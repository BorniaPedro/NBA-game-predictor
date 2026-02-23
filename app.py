import streamlit as st
import pandas as pd
import joblib
from datetime import datetime, timedelta
from nba_api.stats.endpoints import scoreboardv2
from streamlit_extras.stylable_container import stylable_container

# --- Configs ---
st.set_page_config(
    page_title="NBA Game Predictor",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Style
st.markdown("""
    <style>
    .block-container {
        padding-top: 3rem;
        padding-bottom: 2rem;
    }
    
    div[data-testid="column"] > div > div {
        gap: 0.2rem !important;
    }
    
    /* Estilo dos cards */
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px;
        border: 1px solid #e0e0e0;
    }
""", unsafe_allow_html=True)

# --- Data Loading ---
@st.cache_resource
def load_assets():
    """Load model, feature list and historical data"""
    try:
        model = joblib.load('xgb model/nba_model_xgb.pkl')
        features = joblib.load('./xgb model/feature_list.pkl')
            
        history = pd.read_parquet('./dataframes/nba_games_history.parquet')
        history['GAME_DATE'] = pd.to_datetime(history['GAME_DATE'])
        return model, features, history
    except Exception as e:
        st.error(f"ERROR loading the assets: {e}")
        st.stop()

xgb_model, features_xgb, df_history = load_assets()

# --- Preping ID and Matchups ---
@st.cache_data
def create_maps(df):
    # ID Map
    id_map = df[['HOME_TEAM_ABBREVIATION', 'HOME_TEAM_ID']].drop_duplicates().set_index('HOME_TEAM_ABBREVIATION')['HOME_TEAM_ID'].to_dict()
            
    # Matchup Map
    temp_df = df.copy()
    temp_df['MATCHUP_STR'] = temp_df['HOME_TEAM_ABBREVIATION'] + ' vs. ' + temp_df['AWAY_TEAM_ABBREVIATION']
    unique_matchups = sorted(temp_df['MATCHUP_STR'].unique().astype(str))
    matchup_map = {m: i for i, m in enumerate(unique_matchups)}

    return id_map, matchup_map

id_map, matchup_map = create_maps(df_history)

# --- Calculate Current ELO ---
@st.cache_data
def get_current_state(df):
    """Get current ELO ratings and historical ELO for momentum calculation"""
    df = df.sort_values(['GAME_DATE', 'GAME_ID'])
    elo_dict = {}
    team_histories = {}
    last_season_dict = {}
    k_factor = 40
    retention = 0.75

    for _, row in df.iterrows():
        h_id, a_id = row['HOME_TEAM_ID'], row['AWAY_TEAM_ID']
        season, h_win = row['SEASON_ID'], row['HOME_WIN']

        # Current ELO
        elo_h = elo_dict.get(h_id, 1500)
        elo_a = elo_dict.get(a_id, 1500)

        # Soft Reset (first season game)
        if last_season_dict.get(h_id) != season:
            elo_h = (elo_h * retention) + (1500 * (1 - retention))
            last_season_dict[h_id] = season
        if last_season_dict.get(a_id) != season:
            elo_a = (elo_a * retention) + (1500 * (1 - retention))
            last_season_dict[a_id] = season

        # Save ELO history for momentum
        if h_id not in team_histories: team_histories[h_id] = []
        if a_id not in team_histories: team_histories[a_id] = []
        team_histories[h_id].append(elo_h)
        team_histories[a_id].append(elo_a)

        # ELO Update
        prob_h = 1 / (1 + 10 ** ((elo_a - elo_h) / 400))
        new_h = elo_h + k_factor * (h_win - prob_h)
        new_a = elo_a + k_factor * ((1 - h_win) - (1 - prob_h))
        elo_dict[h_id], elo_dict[a_id] = new_h, new_a

    # Add final ELO to histories
    for t in elo_dict:
        if t in team_histories: team_histories[t].append(elo_dict[t])

    return elo_dict, team_histories

current_elos, team_elo_histories = get_current_state(df_history)

# --- Support functions for predictions ---
def get_last_team_game(df, team_id, game_date):
    mask = ((df['HOME_TEAM_ID'] == team_id) | (df['AWAY_TEAM_ID'] == team_id)) & (df['GAME_DATE'] < game_date)
    games = df.loc[mask].sort_values('GAME_DATE')
    if games.empty: return None, None
    last_game = games.iloc[-1]
    side = 'home' if last_game['HOME_TEAM_ID'] == team_id else 'away'
    return last_game, side

def build_feature_row(up_row, df_hist):
    h_id, a_id = up_row['HOME_TEAM_ID'], up_row['AWAY_TEAM_ID']
    date = up_row['GAME_DATE']

    last_h, side_h = get_last_team_game(df_hist, h_id, date)
    last_a, side_a = get_last_team_game(df_hist, a_id, date)

    if last_h is None or last_a is None: return None

    # Helper to get stats
    def get_stat(game, side, col):
        prefix = 'HOME_' if side == 'home' else 'AWAY_'
        return game[f'{prefix}{col}']

    # Date calculations
    rest_h = (date - last_h['GAME_DATE']).days
    rest_a = (date - last_a['GAME_DATE']).days
    b2b_h = 1 if rest_h <= 1 else 0
    b2b_a = 1 if rest_a <= 1 else 0

    # ELO & Momentum
    elo_h = current_elos.get(h_id, 1500)
    elo_a = current_elos.get(a_id, 1500)
    
    def get_mom(tid, curr):
        hist = team_elo_histories.get(tid, [])
        return (curr - hist[-5]) if len(hist) >= 5 else 0
    
    mom_h = get_mom(h_id, elo_h)
    mom_a = get_mom(a_id, elo_a)

    # --- MATCHUP ENCODING ---
    matchup_str = up_row['HOME_TEAM_ABBREVIATION'] + ' vs. ' + up_row['AWAY_TEAM_ABBREVIATION']
    matchup_id = matchup_map.get(matchup_str, -1) 

    # Dictionary
    row = {
        'SEASON_ID': last_h['SEASON_ID'],
        'HOME_TEAM_ID': h_id,
        'AWAY_TEAM_ID': a_id,
        'GAME_DATE': date,
        'MATCHUP': matchup_id,
        'HOME_DAYS_BETWEEN_GAMES': rest_h,
        'AWAY_DAYS_BETWEEN_GAMES': rest_a,
        'HOME_IS_B2B': b2b_h,
        'AWAY_IS_B2B': b2b_a,
        'REST_DIFF': rest_h - rest_a,
        'HOME_ELO': elo_h,
        'AWAY_ELO': elo_a,
        'ELO_DIFF': elo_h - elo_a,
        'HOME_ELO_MOMENTUM': mom_h,
        'AWAY_ELO_MOMENTUM': mom_a,
        'MOMENTUM_DIFF': mom_h - mom_a
    }

    stats_cols = [
        'WINS_LAST_5_GAMES', 'SEASON_RECORD_PCT', 'GAME_COUNT', 
        'ROLLING_PTS_PER_GAME', 'ROLLING_PLUS_MINUS_PER_GAME',
        'SEASON_FG_PCT', 'SEASON_FG3_PCT', 'SEASON_FT_PCT',
        'SEASON_REB_PER_GAME', 'SEASON_TOV_PER_GAME'
    ]
    
    for col in stats_cols:
        row[f'HOME_{col}'] = get_stat(last_h, side_h, col)
        row[f'AWAY_{col}'] = get_stat(last_a, side_a, col)
        
    row['HOME_GAME_COUNT'] += 1
    row['AWAY_GAME_COUNT'] += 1

    return row

# Team color mapping (for visualization)
nba_team_colors = {
    "CHA": {"primary": "#1d1160", "secondary": "#00788C"}, 
    "HOU": {"primary": "#ce1141", "secondary": "#000000"}, 
    "LAL": {"primary": "#552583", "secondary": "#f9a01b"}, 
    "MIA": {"primary": "#98002e", "secondary": "#f9a01b"}, 
    "BOS": {"primary": "#007a33", "secondary": "#BA9653"},
    "LAC": {"primary": "#c8102E", "secondary": "#1d428a"},
    "BKN": {"primary": "#000000", "secondary": "#ffffff"},
    "CHI": {"primary": "#CE1141", "secondary": "#000000"},
    "ATL": {"primary": "#e03a3e", "secondary": "#c1d32f"},
    "PHX": {"primary": "#1d1160", "secondary": "#e56020"},
    "DAL": {"primary": "#00538c", "secondary": "#b8c4ca"},
    "DEN": {"primary": "#0E2240", "secondary": "#fec524"},
    "DET": {"primary": "#c8102e", "secondary": "#1d42ba"},
    "GSW": {"primary": "#1d428a", "secondary": "#ffc72c"},
    "IND": {"primary": "#002d62", "secondary": "#fdbb30"},
    "MEM": {"primary": "#5d76a9", "secondary": "#12173f"},
    "MIL": {"primary": "#00471b", "secondary": "#eee1c6"},
    "MIN": {"primary": "#0c2340", "secondary": "#236192"},
    "NOP": {"primary": "#0c2340", "secondary": "#85714d"},
    "NYK": {"primary": "#006bb6", "secondary": "#f58426"},
    "OKC": {"primary": "#007ac1", "secondary": "#ef3b24"},
    "ORL": {"primary": "#0077c0", "secondary": "#c4ced4"},
    "PHI": {"primary": "#006bb6", "secondary": "#ed174c"},
    "POR": {"primary": "#e03a3e", "secondary": "#000000"},
    "SAC": {"primary": "#5a2d81", "secondary": "#63727a"},
    "SAS": {"primary": "#c4ced4", "secondary": "#000000"},
    "TOR": {"primary": "#ce1141", "secondary": "#000000"},
    "UTA": {"primary": "#753bbd", "secondary": "#ffffff"},
    "CLE": {"primary": "#860038", "secondary": "#fdbb30"},
    "WAS": {"primary": "#002b5c", "secondary": "#e31837"},
}

def get_team_badge(team_code):
    colors = nba_team_colors.get(team_code, {"primary": "#808080", "secondary": "#000000"})
    p, s = colors['primary'], colors['secondary']
    
    # Return a circular badge with a gradient of primary and secondary colors
    return f"""<div style="display: inline-block; width: 24px; height: 24px; border-radius: 50%; background: conic-gradient(from 0deg, {s} 0deg 180deg, {p} 180deg 360deg); box-shadow: inset 0 0 0 2px rgba(255,255,255,0.2); margin: 0 8px; vertical-align: middle;"></div>"""

# Nicknames
NBA_NICKNAMES = {
    'ATL': 'Hawks', 'BOS': 'Celtics', 'BKN': 'Nets', 'CHA': 'Hornets',
    'CHI': 'Bulls', 'CLE': 'Cavaliers', 'DAL': 'Mavericks', 'DEN': 'Nuggets',
    'DET': 'Pistons', 'GSW': 'Warriors', 'HOU': 'Rockets', 'IND': 'Pacers',
    'LAC': 'Clippers', 'LAL': 'Lakers', 'MEM': 'Grizzlies', 'MIA': 'Heat',
    'MIL': 'Bucks', 'MIN': 'Timberwolves', 'NOP': 'Pelicans', 'NYK': 'Knicks',
    'OKC': 'Thunder', 'ORL': 'Magic', 'PHI': '76ers', 'PHX': 'Suns',
    'POR': 'Trail Blazers', 'SAC': 'Kings', 'SAS': 'Spurs', 'TOR': 'Raptors',
    'UTA': 'Jazz', 'WAS': 'Wizards'
}

def get_nickname(nickname):
    return NBA_NICKNAMES.get(nickname, nickname)

def get_team_card(team_name, role, align="center"):
    # Define alignment
    flex_align = "center"
    text_align = "center"
    badge = get_team_badge(team_name)

    content = f'<span style="font-size: 26px; font-weight: 800; color: white; line-height: 1;">{get_nickname(team_name)}</span>{badge}'

    return f"""
    <div style="display: flex; align-items: center; justify-content: {flex_align}; width: 100%; min-height: 80px;">
        <div style="display: flex; align-items: center; justify-content: center;">
            {content}
        </div>
    </div>
    """

# --- Dashboard logic ---

# Sidebar
with st.sidebar:
    st.header("Settings")
    target_date = st.date_input("Game Date", datetime.now())
    st.info("The model uses historical data up to today to calculate fatigue and ELO.")
    st.markdown("---")
    st.caption(f"Model used: XGBoost")
    st.caption(f"History contains {len(df_history)} games")
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: gray; font-size: 0.8rem;">
            <strong>Legal Advisory</strong><br>
            This project uses NBA and team names for educational purposes under fair use. It is not affiliated with, sponsored by, or endorsed by the National Basketball Association. This site does not promote gambling; please play responsibly.
        </div>
        """
        , unsafe_allow_html=True
    )

# Botão Principal
with stylable_container(
    "Make Predictions",
    css_styles="""
    button[kind="primary"] {
        background-color: #262730 !important;
        border: 1px solid #888 !important;
        color: white !important;
    }
    button[kind="primary"]:hover {
        background-color: #0e1117 !important;
        border-color: #888 !important;
    }""",
):
    if st.button("Make Predictions", type="primary", icon=":material/wand_stars:"):
        with st.spinner(f"Finding games for {target_date.strftime('%d/%m/%Y')}..."):
            
            # API Call
            date_str = target_date.strftime('%Y-%m-%d')
            try:
                board = scoreboardv2.ScoreboardV2(game_date=date_str, day_offset=0)
                df_games = board.game_header.get_data_frame()
                line_score = board.line_score.get_data_frame()
            except Exception as e:
                st.error(f"ERROR connecting to the NBA API: {e}")
                st.stop()
            
            if df_games.empty:
                st.warning("No games found for this date.")
            else:
                df_games = df_games.drop_duplicates(subset=['GAME_ID'])

                # Order by time
                if 'GAME_SEQUENCE' in df_games.columns:
                    df_games = df_games.sort_values(by='GAME_SEQUENCE')

                line_score = line_score.drop_duplicates(subset=['GAME_ID', 'TEAM_ID'])

                # Merge to get team abbreviations
                df_games = pd.merge(df_games, line_score[['GAME_ID', 'TEAM_ID', 'TEAM_ABBREVIATION']], 
                                    left_on=['GAME_ID', 'HOME_TEAM_ID'], right_on=['GAME_ID', 'TEAM_ID'], how='left')
                df_games.rename(columns={'TEAM_ABBREVIATION': 'HOME_TEAM_ABBREVIATION'}, inplace=True)
                df_games.drop(columns=['TEAM_ID'], inplace=True)
                
                df_games = pd.merge(df_games, line_score[['GAME_ID', 'TEAM_ID', 'TEAM_ABBREVIATION']], 
                                    left_on=['GAME_ID', 'VISITOR_TEAM_ID'], right_on=['GAME_ID', 'TEAM_ID'], how='left')
                df_games.rename(columns={'TEAM_ABBREVIATION': 'AWAY_TEAM_ABBREVIATION'}, inplace=True)
                df_games.drop(columns=['TEAM_ID'], inplace=True)
                
                # Map IDs
                df_games['HOME_TEAM_ID'] = df_games['HOME_TEAM_ABBREVIATION'].map(id_map)
                df_games['AWAY_TEAM_ID'] = df_games['AWAY_TEAM_ABBREVIATION'].map(id_map)
                df_games = df_games.dropna(subset=['HOME_TEAM_ID', 'AWAY_TEAM_ID'])
                
                if df_games.empty:
                    st.error("ERROR: Could not map teams (IDs not found).")
                    st.stop()
                    
                df_games['HOME_TEAM_ID'] = df_games['HOME_TEAM_ID'].astype(int)
                df_games['AWAY_TEAM_ID'] = df_games['AWAY_TEAM_ID'].astype(int)
                df_games['GAME_DATE'] = pd.to_datetime(df_games['GAME_DATE_EST'])
                
                # Build features for each game
                pred_rows = []
                valid_indices = []
                
                for idx, row in df_games.iterrows():
                    feat = build_feature_row(row, df_history)
                    if feat:
                        pred_rows.append(feat)
                        valid_indices.append(idx)
                
                if pred_rows:
                    X_pred = pd.DataFrame(pred_rows)
                    
                    # Align features with model
                    booster = xgb_model.get_booster()
                    model_features = booster.feature_names
                    
                    # Fill missing features with 0
                    for col in model_features:
                        if col not in X_pred.columns:
                            X_pred[col] = 0 
                    X_pred = X_pred[model_features]
                    
                    # Predict
                    probs = xgb_model.predict_proba(X_pred)[:, 1]
                    
                    # Display results
                    st.subheader(f"Results for {target_date.strftime('%d/%m/%Y')}")

                    # Headers
                    h1, h2, h3, h4 = st.columns([1.5, 2, 1.5, 1])
                    with h1:
                        st.markdown("<div style='text-align: center; color: #888; font-weight: bold; font-size: 15px; letter-spacing: 1px;'>VISITOR TEAM</div>", unsafe_allow_html=True)
                    with h2:
                        st.markdown("<div style='text-align: center; color: #888; font-weight: bold; font-size: 15px; letter-spacing: 1px;'>TIME & WIN PROBABILITY</div>", unsafe_allow_html=True)
                    with h3:
                        st.markdown("<div style='text-align: center; color: #888; font-weight: bold; font-size: 15px; letter-spacing: 1px;'>HOME TEAM</div>", unsafe_allow_html=True)
                    with h4:
                        st.markdown("<div style='text-align: center; color: #888; font-weight: bold; font-size: 15px; letter-spacing: 1px;'>PREDICTION</div>", unsafe_allow_html=True)

                    st.divider()
                    
                    # Layout in cards
                    for i, prob_home in enumerate(probs):
                        # Original data
                        orig_row = df_games.loc[valid_indices[i]]
                        home = orig_row['HOME_TEAM_ABBREVIATION']
                        away = orig_row['AWAY_TEAM_ABBREVIATION']
                        game_time = orig_row['GAME_STATUS_TEXT']
                        
                        prob_away = 1 - prob_home
                        winner = home if prob_home > 0.5 else away
                        conf = max(prob_home, prob_away)
                        
                        # Style based on confidence
                        if conf > 0.65:
                            color = "green"
                            icon = "🔥"
                            status = "High Confidence"
                        elif conf > 0.55:
                            color = "orange"
                            icon = "⚖️"
                            status = "Balanced"
                        else:
                            color = "gray"
                            icon = "🎲"
                            status = "Uncertain"

                        # Visual container
                        with st.container():
                            c1, c2, c3, c4 = st.columns([1.5, 2, 1.5, 1])
                            with c1:
                                st.markdown(get_team_card(away, "Visitor", align="center"), unsafe_allow_html=True)
                            with c2:
                                st.markdown(f"<div style='text-align: center; width: 100%; margin-bottom: 5px;'><span style='font-weight: bold; color: #white; background: #262730; border-radius: 4px; padding: 2px 10px; font-size: 12px;'>{game_time}</span></div>", unsafe_allow_html=True)
                                st.markdown(f"<div style='text-align: center; width: 100%;'>", unsafe_allow_html=True)
                                st.progress(float(prob_away))
                                st.markdown(f"</div><div style='text-align: center; color: #888; font-size: 13px;'>{prob_away:.1%} vs {prob_home:.1%}</div>", unsafe_allow_html=True)
                            with c3:
                                st.markdown(get_team_card(home, "Home", align="center"), unsafe_allow_html=True)
                            with c4:
                                st.markdown(
                                    f"""
                                    <div style="background-color: rgba(255,255,255,0.05); padding: 8px; border-radius: 8px; text-align: center;">
                                        <div style="font-size: 11px; color: #aaa; text-transform: uppercase;">Winner</div>
                                        <div style="font-weight: bold; font-size: 18px; color: {color};">{get_nickname(winner)}</div>
                                        <div style="font-size: 18px;">{icon}</div>
                                    </div>
                                    """, 
                                    unsafe_allow_html=True
                                )
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                            st.markdown('<div style="height: 1px; background-color: #333; width: 100%;"></div>', unsafe_allow_html=True)
                    
                    # Detailed table
                    with st.expander("View Detailed Table"):
                        resumo = pd.DataFrame({
                            'Home Team': df_games.loc[valid_indices, 'HOME_TEAM_ABBREVIATION'],
                            'Away Team': df_games.loc[valid_indices, 'AWAY_TEAM_ABBREVIATION'],
                            'Home Prob': probs,
                            'Away Prob': 1-probs,
                            'Confidence': [max(p, 1-p) for p in probs]
                        })
                        st.dataframe(resumo.style.format({'Home Prob': '{:.1%}', 'Away Prob': '{:.1%}', 'Confidence': '{:.1%}'})
                                    .background_gradient(subset=['Confidence'], cmap='Greens'))
                        
                else:
                    st.warning("It was not possible to build features for any game (missing historical data).")

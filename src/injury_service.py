import requests
from datetime import datetime

# Mapping of ESPN team abbreviations to NBA API standard abbreviations
ESPN_TO_NBA_TEAMS = {
    "GS": "GSW",
    "NY": "NYK",
    "SA": "SAS",
    "UTAH": "UTA",
    "WSH": "WAS",
    "NO": "NOP",
    "PHO": "PHX",
    "BK": "BKN",
    # Direct mappings for other teams:
    "ATL": "ATL", "BOS": "BOS", "CHA": "CHA", "CHI": "CHI", 
    "CLE": "CLE", "DAL": "DAL", "DEN": "DEN", "DET": "DET", 
    "HOU": "HOU", "IND": "IND", "LAC": "LAC", "LAL": "LAL", 
    "MEM": "MEM", "MIA": "MIA", "MIL": "MIL", "MIN": "MIN", 
    "OKC": "OKC", "ORL": "ORL", "PHI": "PHI", "POR": "POR", 
    "SAC": "SAC", "TOR": "TOR"
}

def normalize_team_abbr(espn_abbr):
    """
    Guarantees that the ESPN team abbreviation is converted to the standard NBA API abbreviation.
    """
    espn_abbr = espn_abbr.upper().strip()
    return ESPN_TO_NBA_TEAMS.get(espn_abbr, espn_abbr)

def get_nba_injuries_espn(target_date=None):
    """
    Fetches NBA injury data from ESPN for a given date.
    
    :param target_date: Object datetime.date, datetime.datetime or string 'YYYY-MM-DD' / 'YYYYMMDD'.
                        If None, defaults to today's date.
    :return: Dictionary indexed by team abbreviation (NBA API standard) with a list of injuries:
             {
               'GSW': [{'name': 'Stephen Curry', 'status': 'OUT', 'p_out': 1.0, 'detail': 'Ankle'}],
               ...
             }
    """
    # Treat the target_date input to ensure it's in the correct format for the ESPN API
    if target_date is None:
        date_str = datetime.today().strftime('%Y%m%d')
    elif isinstance(target_date, (datetime, )):
        date_str = target_date.strftime('%Y%m%d')
    elif isinstance(target_date, str):
        date_str = target_date.replace('-', '').strip()
    else:
        date_str = str(target_date).replace('-', '').strip()

    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_str}"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            print(f"Warning: ESPN API returned status code {response.status_code}")
            return {}
        data = response.json()
    except Exception as e:
        print(f"Error connecting to ESPN API: {e}")
        return {}

    team_injuries = {}

    # OUT chances based on ESPN injury status text
    status_weights = {
        "OUT": 1.0,
        "DOUBTFUL": 0.75,
        "QUESTIONABLE": 0.50,
        "DAY-TO-DAY": 0.40,
        "PROBABLE": 0.15
    }

    for event in data.get("events", []):
        for comp in event.get("competitions", []):
            for competitor in comp.get("competitors", []):
                raw_abbr = competitor.get("team", {}).get("abbreviation", "")
                team_abbr = normalize_team_abbr(raw_abbr)
                
                injuries = competitor.get("injuries", [])
                
                if team_abbr not in team_injuries:
                    team_injuries[team_abbr] = []

                for inj in injuries:
                    athlete = inj.get("athlete", {})
                    athlete_name = athlete.get("displayName")
                    status_text = inj.get("status", "").upper()
                    detail_text = inj.get("details", {}).get("type", "")

                    # Determine the probability of being out based on the status text
                    p_out = 0.0
                    for key, val in status_weights.items():
                        if key in status_text:
                            p_out = val
                            break
                    
                    # Handle cases where the status text indicates "OUT" but wasn't captured in the weights
                    if p_out == 0.0 and "OUT" in status_text:
                        p_out = 1.0

                    # Only include injuries with a non-zero probability of being out and a valid athlete name
                    if p_out > 0.0 and athlete_name:
                        team_injuries[team_abbr].append({
                            "name": athlete_name,
                            "status": status_text,
                            "p_out": p_out,
                            "detail": detail_text
                        })

    return team_injuries
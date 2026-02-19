# NBA Game Predictor

A machine learning project that predicts NBA game outcomes using historical data, ELO ratings, and team statistics. Built with XGBoost and Streamlit.

## Live Demo

**Access the app:** [https://borniapedro-nba-game-predictor.streamlit.app/](https://borniapedro-nba-game-predictor.streamlit.app/)

## Overview

This project uses machine learning to predict NBA game winners by analyzing:
- ELO ratings and momentum
- Team fatigue (days between games, back-to-back games)
- Season and rolling statistics (points, FG%, rebounds, turnovers)
- Recent form (last 5 games)

The model is trained on 10,000+ regular season games from 2019 to present and achieves ~64% accuracy.

## Installation

```bash
git clone https://github.com/BorniaPedro/NBA-game-predictor.git
cd NBA-game-predictor
pip install -r requirements.txt
streamlit run app.py
```

## Usage

**Online:** Visit the [live demo](https://borniapedro-nba-game-predictor.streamlit.app/)

**Locally:**
1. Run `streamlit run app.py`
2. Select a game date
3. Click "Make Predictions"
4. View predictions with confidence levels:
   - 🔥 High Confidence (>65%)
   - ⚖️ Balanced (55-65%)
   - 🎲 Uncertain (<55%)

## Project Structure

```
NBA-game-predictor/
├── app.py                              # Streamlit web app
├── requirements.txt                    # Dependencies
├── notebooks/
│   ├── game_predictor.ipynb           # Model training
│   └── next_game_predictor.ipynb      # Prediction pipeline
├── dataframes/
│   ├── nba_games_2017_today.parquet   # Raw data
│   └── nba_games_history.parquet      # Processed data
└── xgb model/
    ├── nba_model_xgb.pkl              # Trained model
    └── feature_list.pkl               # Features
```

## Technologies

- **Streamlit** - Web interface
- **XGBoost** - Machine learning model
- **pandas** - Data processing
- **nba_api** - NBA data source
- **scikit-learn** - Model evaluation

## Model Details

- **Algorithm:** XGBoost Classifier
- **Training data:** 10,000+ games (2019-present)
- **Features:** 30+ engineered features
- **Accuracy:** ~64%
- **Data source:** NBA API (regular season only)

## Legal Notice

This project is for educational purposes only. Not affiliated with the NBA. Does not promote gambling.

## Author

Pedro Bornia - [@BorniaPedro](https://github.com/BorniaPedro)
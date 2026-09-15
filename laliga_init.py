from pandas import read_csv
import pandas as pd

df = read_csv('LaLiga_Matches.csv')

season = input('Enter the season: ')
team = input('Enter the team: ')

df2023 =df[df['Season'] == season]

rm2023 = df2023[df2023['HomeTeam'] ==  team]
rm2023['result'] = rm2023.apply(lambda x: 'W' if x['FTHG'] > x['FTAG'] else 'D' if x['FTHG'] == x['FTAG'] else 'L', axis=1)
rm2023_away = df2023[df2023['AwayTeam'] == team]
rm2023_away['result'] = rm2023_away.apply(lambda x: 'W' if x['FTHG'] < x['FTAG'] else 'D' if x['FTHG'] == x['FTAG'] else 'L', axis=1)

rm2023 = pd.concat([rm2023, rm2023_away], sort=True)

print(rm2023)


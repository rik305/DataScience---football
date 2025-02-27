import requests, json
from pprint import pprint
import pandas as pd
import numpy as np

base_url = 'https://fantasy.premierleague.com/api/'

r = requests.get(base_url+'bootstrap-static/').json()

players = r['elements']
teams = r['teams']

players = pd.json_normalize(players)
teams = pd.json_normalize(teams)


players.set_index('id', inplace=True)
teams.set_index('id', inplace=True)

# Constants for positions
GOALKEEPER = 1
DEFENDER = 2
MIDFIELDER = 3
FORWARD = 4

# Define your budget and number of players needed per position
budget = 1000  
team_size = 15
gk_size = 2
def_size = 5
mid_size = 5
fwd_size = 3
max_per_team = 3

# Create empty lists to hold players in each position
goalkeepers = []
defenders = []
midfielders = []
forwards = []

players.merge(teams, left_on='team', right_index=True, suffixes=('', '_team'))
players = players[["first_name","second_name","total_points","points_per_game","form","now_cost","selected_by_percent","chance_of_playing_this_round","element_type","team"]]

players['name'] = players["first_name"] + " " + players["second_name"]

players.drop(['first_name', 'second_name'], axis=1, inplace=True)
players = players[players["total_points"]>players["total_points"].mean()]
players["ppg"] = pd.to_numeric(players["points_per_game"], errors='coerce')
players.drop(['points_per_game'], axis=1, inplace=True)
players = players[players["ppg"]>players["ppg"].mean()]
#players = players[players["chance_of_playing_this_round"]>80]
players["form"] = pd.to_numeric(players["form"], errors='coerce')
players = players[players["form"] > players["form"].mean()]
players.set_index('name', inplace=True)
players['count'] = 0

for i, player in players.iterrows():
   if player['element_type'] == GOALKEEPER:
        goalkeepers.append(player)
   elif player['element_type'] == DEFENDER:
       defenders.append(player)
   elif player['element_type'] == MIDFIELDER:
       midfielders.append(player)
   elif player['element_type'] == FORWARD:
       forwards.append(player)


gk = pd.DataFrame(goalkeepers)
df = pd.DataFrame(defenders)
md = pd.DataFrame(midfielders)
fw = pd.DataFrame(forwards)

for i in ["total_points","ppg","form","selected_by_percent"]:
    gk.sort_values(by=i, ascending=False, inplace=True)
    df.sort_values(by=i, ascending=False, inplace=True)
    md.sort_values(by=i, ascending=False, inplace=True)
    fw.sort_values(by=i, ascending=False, inplace=True)
    for a in range(len(gk)):
        gk.loc[gk.index[a], 'count'] += a
    for a in range(len(df)):
        df.loc[df.index[a], 'count'] += a
    for a in range(len(md)):
        md.loc[md.index[a], 'count'] += a
    for a in range(len(fw)):
        fw.loc[fw.index[a], 'count'] += a
        

gk.sort_values(by='count', ascending=True, inplace=True)
df.sort_values(by='count', ascending=True, inplace=True)
md.sort_values(by='count', ascending=True, inplace=True)
fw.sort_values(by='count', ascending=True, inplace=True)
        
players = pd.concat([gk, df, md, fw])
players.sort_values(by=['count', 'selected_by_percent'], ascending=[True, False], inplace=True)

positions=[0,0,0,0,0]
teams = [0] * 21
team = []
cost = 0

i = 0
c = 0

while len(team) != 5:
    player = players.iloc[i]
    if (positions[player['element_type']-1] < [0,gk_size, def_size, mid_size, fwd_size][player['element_type']] 
    and teams[player['team']] < max_per_team):
        team.append([player.name, player['element_type']])
        cost += player['now_cost']
        positions[player['element_type']-1] += 1
        teams[player['team']] += 1
        c+=1
        i+=1

budget -= cost
while len(team) != 15:
    pavg = budget/(16-c)
    player = players.iloc[i]
    if (positions[player['element_type']-1] < [0,gk_size, def_size, mid_size, fwd_size][player['element_type']] 
    and teams[player['team']] < max_per_team and pavg>=player['now_cost']):
        team.append([player.name, player['element_type']])
        cost += player['now_cost']
        budget -= player['now_cost']
        positions[player['element_type']-1] += 1
        teams[player['team']] += 1
        c += 1
        
    i += 1

team = sorted(team, key=lambda x: x[1])

for i in team:
    print (players.loc[i[0]])

for i in [0,2,3,4,7,8,9,10,11,12,13,14]:
    print (players.loc[team[i][0]])
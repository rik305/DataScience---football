from __future__ import print_function
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


# Constants for positions
GOALKEEPER = 1
DEFENDER = 2
MIDFIELDER = 3
FORWARD = 4

# Define your budget and number of players needed per position
budget = 1040 
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

for i in players.columns:
    print(i)

players = players[["first_name","second_name","total_points","points_per_game","form","now_cost","selected_by_percent","chance_of_playing_this_round","element_type","team","id"]]

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

constraints = ["ppg","form","total_points","selected_by_percent"]
for i in range(len(constraints)):
    gk.sort_values(by=constraints[i], ascending=False, inplace=True)
    df.sort_values(by=constraints[i], ascending=False, inplace=True)
    md.sort_values(by=constraints[i], ascending=False, inplace=True)
    fw.sort_values(by=constraints[i], ascending=False, inplace=True)
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

while len(team) != 3:
    player = players.iloc[i]
    if (positions[player['element_type']-1] < [0,gk_size, def_size, mid_size, fwd_size][player['element_type']] 
    and teams[player['team']] < max_per_team):
        team.append([player.name, player['element_type'],player['id']])
        cost += player['now_cost']
        positions[player['element_type']-1] += 1
        teams[player['team']] += 1
        c+=1
        i+=1 

budget -= cost
s = 2
while len(team) != 15:
    pavg = budget/(15-c)
    player = players.iloc[i]
    if (positions[player['element_type']-1] < [0,gk_size, def_size, mid_size, fwd_size][player['element_type']] 
    and teams[player['team']] < max_per_team and pavg + s>=player['now_cost'] and player.name not in [x[0] for x in team]):
        team.append([player.name, player['element_type'],player['id']])
        cost += player['now_cost']
        budget -= player['now_cost']
        positions[player['element_type']-1] += 1
        teams[player['team']] += 1
        c += 1
        i = 3
        s *= -1
        
    i += 1

team = sorted(team, key=lambda x: 1000*x[1] + players.loc[x[0]]['count'])
print("----------------------------------")
print("Team:")
for i in team:
    print (i[0], players.loc[i[0]]['count'])
    print (players.loc[i[0]]['total_points'])

print("Total Cost:", cost)
print("Remaining Budget:", budget)

tp = 0
def tp_352(team):
    tp = 0
    s = ""
    print(("3-5-2:").center(100))
    print("\n")
    for i in [0,2,3,4,7,8,9,10,11,12,13]:
        name = team[i][0].split()
        s+=name[0][0]+". "+name[-1]+"  "
        if (i in [0,4,11,13]):
            print(s.center(100), "\n")
            s = ''
        tp += players.loc[team[i][0]]['total_points']
    print(("Total Points: " + (str(tp))).center(100))
    print("\n")
    return tp

def tp_442(team):
    tp = 0
    s = ""
    print(("4-4-2:").center(100))
    print("\n")
    for i in [0,2,3,4,5,7,8,9,10,12,13]:
        name = team[i][0].split()
        s+=name[0][0]+". "+name[-1]+"  "
        if (i in [0,5,10,13]):
            print(s.center(100), "\n")
            s = ''
        tp += players.loc[team[i][0]]['total_points']
    print(("Total Points: " + (str(tp))).center(100))
    print("\n")
    return tp



def tp_343(team):
    tp = 0
    s = ""
    print(("3-4-3:").center(100))
    print("\n")
    for i in [0,2,3,4,7,8,9,10,12,13,14]:
        name = team[i][0].split()
        s+=name[0][0]+". "+name[-1]+"  "
        if (i in [0,4,10,14]):
            print(s.center(100), "\n")
            s = ''
        tp += players.loc[team[i][0]]['total_points']
    print(("Total Points: " + (str(tp))).center(100))
    print("\n")
    return tp

def get_gameweek_history(player_id):
    r = requests.get(
            base_url + 'element-summary/' + str(player_id) + '/'
    ).json()
    df = pd.json_normalize(r['history'])
    return df 

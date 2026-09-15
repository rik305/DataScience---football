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

t = [0]
for i in range(20):
    t.append(teams.loc[i]['name'])

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
players.head(10)

for col in ['expected_goals', 'expected_assists', 'expected_goals_conceded', 'minutes']:
    players[col] = pd.to_numeric(players[col], errors='coerce').fillna(0)

team_attack = players.groupby('team')[['expected_goals', 'expected_assists']].sum()
teams['attack'] = team_attack['expected_goals'] + team_attack['expected_assists']
gk_xgc = players[players['element_type'] == GOALKEEPER].groupby('team')['expected_goals_conceded'].sum()
teams['defense'] = -gk_xgc.reindex(teams.index).fillna(gk_xgc.mean())
venue_mean = teams[['strength_overall_home', 'strength_overall_away']].replace(0, np.nan).mean().mean()
if pd.isna(venue_mean) or venue_mean == 0:
    venue_mean = 3.0
teams['attack_home'] = teams['attack'] * (teams['strength_overall_home'].replace(0, venue_mean) / venue_mean)
teams['attack_away'] = teams['attack'] * (teams['strength_overall_away'].replace(0, venue_mean) / venue_mean)
teams['defence_home'] = teams['defense'] * (teams['strength_overall_home'].replace(0, venue_mean) / venue_mean)
teams['defence_away'] = teams['defense'] * (teams['strength_overall_away'].replace(0, venue_mean) / venue_mean)

players = players[["first_name","second_name","total_points","points_per_game","form","now_cost","selected_by_percent","chance_of_playing_this_round","element_type","team","id","influence","expected_goals_per_90","expected_assists_per_90","goals_scored","assists",]]
players['name'] = players["first_name"] + " " + players["second_name"]
players.set_index('name', inplace=True)

players.drop(['first_name', 'second_name'], axis=1, inplace=True)
#players = players[players["total_points"]>players["total_points"].mean()]
players["ppg"] = pd.to_numeric(players["points_per_game"], errors='coerce')
players.drop(['points_per_game'], axis=1, inplace=True)
players = players[players["ppg"]>2]
                  #players["ppg"].mean()]
#players = players[players["chance_of_playing_this_round"]>80]
players["form"] = pd.to_numeric(players["form"], errors='coerce')
players = players[players["form"] > 1]
                  #players["form"].mean()]
players['count'] = 0
players['expected_goals_per_90'] = pd.to_numeric(players['expected_goals_per_90'], errors='coerce').fillna(0)
players['expected_assists_per_90'] = pd.to_numeric(players['expected_assists_per_90'], errors='coerce').fillna(0)
players['goals_scored'] = pd.to_numeric(players['goals_scored'], errors='coerce').fillna(0)
players['assists'] = pd.to_numeric(players['assists'], errors='coerce').fillna(0)


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

FDR_LOOKAHEAD = 5
DEF_GA_FULL_WEIGHT = 8.0
DEF_XGI90_FULL_WEIGHT = 0.20

def _fixture_strengths(fixtures_df):
    fixtures_df = fixtures_df.copy()
    fixtures_df['home_attack'] = fixtures_df['team_h'].map(teams['attack_home'])
    fixtures_df['home_defense'] = fixtures_df['team_h'].map(teams['defence_home'])
    fixtures_df['away_attack'] = fixtures_df['team_a'].map(teams['attack_away'])
    fixtures_df['away_defense'] = fixtures_df['team_a'].map(teams['defence_away'])
    return fixtures_df

def _own_opp_strengths(row, is_home):
    if is_home:
        return (
            row['home_attack'],
            row['home_defense'],
            row['away_attack'],
            row['away_defense'],
        )
    return (
        row['away_attack'],
        row['away_defense'],
        row['home_attack'],
        row['home_defense'],
    )

def _defender_attack_weight(goals, assists, xg90, xa90):
    ga_weight = min(1.0, (float(goals) + float(assists)) / DEF_GA_FULL_WEIGHT)
    xgi_weight = min(1.0, (float(xg90) + float(xa90)) / DEF_XGI90_FULL_WEIGHT)
    return max(ga_weight, xgi_weight)

def _upcoming_fixtures():
    fixtures = requests.get(base_url+'fixtures/').json()
    fixtures_df = pd.DataFrame(fixtures)
    fixtures_df = fixtures_df[fixtures_df['finished'] == False]
    fixtures_df = fixtures_df.sort_values(by=['event', 'kickoff_time'], na_position='last')
    return _fixture_strengths(fixtures_df)

upcoming_fixtures = _upcoming_fixtures()

def add_fdr(player_df, fixtures_df=upcoming_fixtures, lookahead=FDR_LOOKAHEAD):
    if player_df.empty:
        player_df = player_df.copy()
        player_df['fdr'] = pd.Series(dtype=float)
        return player_df

    scores = []
    for _, player in player_df.iterrows():
        team_id = player['team']
        team_fixtures = fixtures_df[
            (fixtures_df['team_h'] == team_id) | (fixtures_df['team_a'] == team_id)
        ].head(lookahead)

        fixture_scores = []
        for _, fixture in team_fixtures.iterrows():
            is_home = fixture['team_h'] == team_id
            own_att, own_def, opp_att, opp_def = _own_opp_strengths(fixture, is_home)
            attacking_fdr = own_att - opp_def
            defending_fdr = own_def - opp_att

            if player['element_type'] in (MIDFIELDER, FORWARD):
                fixture_scores.append(attacking_fdr)
            elif player['element_type'] == DEFENDER:
                weight = _defender_attack_weight(
                    player.get('goals_scored', 0),
                    player.get('assists', 0),
                    player.get('expected_goals_per_90', 0),
                    player.get('expected_assists_per_90', 0),
                )
                fixture_scores.append(defending_fdr + weight * attacking_fdr)
            else:
                fixture_scores.append(defending_fdr)

        scores.append(np.mean(fixture_scores) if fixture_scores else 0.0)

    player_df = player_df.copy()
    player_df['fdr'] = scores
    return player_df

gk = add_fdr(gk)
df = add_fdr(df)
md = add_fdr(md)
fw = add_fdr(fw)

constraints = ["ppg","form","selected_by_percent", 'total_points', 'fdr']
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
players.sort_values(by=['count', 'ppg'], ascending=[True, False], inplace=True)


positions=[0,0,0,0,0]
iteams = [0] * 21
team = []
cost = 0

i = 0
c = 0

while len(team) != 3:
    player = players.iloc[i]
    if (positions[player['element_type']-1] < [0,gk_size, def_size, mid_size, fwd_size][player['element_type']] 
    and iteams[player['team']] < max_per_team):
        team.append([player.name, player['element_type'],player['id']])
        cost += player['now_cost']
        positions[player['element_type']-1] += 1
        iteams[player['team']] += 1
        c+=1
        i+=1 

budget -= cost
s = 5

while len(team) != 15:
    pavg = budget/(15-c)
    player = players.iloc[i]
    if (positions[player['element_type']-1] < [0,gk_size, def_size, mid_size, fwd_size][player['element_type']] 
    and iteams[player['team']] < max_per_team and pavg + s>=player['now_cost'] and player.name not in [x[0] for x in team]):
        team.append([player.name, player['element_type'],player['id']])
        cost += player['now_cost']
        budget -= player['now_cost']
        positions[player['element_type']-1] += 1
        iteams[player['team']] += 1
        c += 1
        i = 2
        s *= -1
        
    i += 1

team = sorted(team, key=lambda x: 1000*x[1] + players.loc[x[0]]['count'])

def print_team(team):
    print("----------------------------------")
    print("Team:")
    for i in team:
        print (i[0], players.loc[i[0]]['count'])
        print (players.loc[i[0]]['total_points'])
        print (players.loc[i[0]]['now_cost']/10)
        print (t[(players.loc[i[0]]['team'])])

    print("Total Cost:", cost)
    print("Remaining Budget:", budget)
 
def tp_352(team):
    tp = 0
    s = ""
    print(("3-5-2:").center(100))
    print("\n")
    for i in [0,2,3,4,7,8,9,10,11,12,13]:
        name = team[i][0].split()
        s+=name[0][0]+". "+name[-1]+"  ""("+t[(players.loc[team[i][0]]['team'])]+")   "
        if (i in [0,4,11,13]):
            print(s.center(100), "\n")
            s = ''
        tp += players.loc[team[i][0]]['total_points']
    tp+=players.iloc[0]['total_points']
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
        s+=name[0][0]+". "+name[-1]+"  ""("+t[(players.loc[team[i][0]]['team'])]+")"
        if (i in [0,5,10,13]):
            print(s.center(100), "\n")
            s = ''
        tp += players.loc[team[i][0]]['total_points']
    tp+=players.iloc[0]['total_points']
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
        s+=name[0][0]+". "+name[-1]+"  " +"("+t[(players.loc[team[i][0]]['team'])]+")"
        if (i in [0,4,10,14]):
            print(s.center(100), "\n")
            s = ''
        tp += players.loc[team[i][0]]['total_points']
    tp+=players.iloc[0]['total_points']
    print(("Total Points: " + (str(tp))).center(100))
    print("\n")
    return tp

def get_gameweek_history(player_id):
    r = requests.get(
            base_url + 'element-summary/' + str(player_id) + '/'
    ).json()
    df = pd.json_normalize(r['history'])
    return df


print(gk.head(7)[['ppg', 'form', 'fdr', 'count']])
print(df.head(7)[['ppg', 'form', 'fdr', 'count']])
print(md.head(7)[['ppg', 'form', 'fdr', 'count']])
print(fw.head(7)[['ppg', 'form', 'fdr', 'count']])


print_team(team)
tp_352(team)
from __future__ import print_function
import requests, json
from pprint import pprint
import pandas as pd
import numpy as np

base_url = 'https://fantasy.premierleague.com/api/'

# Constants for positions
GOALKEEPER = 1
DEFENDER = 2
MIDFIELDER = 3
FORWARD = 4

FDR_LOOKAHEAD = 5
DEF_GA_FULL_WEIGHT = 8.0
DEF_XGI90_FULL_WEIGHT = 0.20

RANK_CONSTRAINTS = ["ppg", "form", "selected_by_percent", "total_points", "fdr"]


def fetch_bootstrap():
    """Fetch the raw bootstrap-static payload (players + teams) from the FPL API."""
    return requests.get(base_url + 'bootstrap-static/').json()


def build_teams(r):
    """Build an enriched teams DataFrame (attack/defense strength, home/away splits)."""
    players_raw = pd.json_normalize(r['elements'])
    teams = pd.json_normalize(r['teams'])

    team_names = [0]
    for i in range(len(teams)):
        team_names.append(teams.loc[i]['name'])

    teams.set_index('id', inplace=True)

    for col in ['expected_goals', 'expected_assists', 'expected_goals_conceded', 'minutes']:
        players_raw[col] = pd.to_numeric(players_raw[col], errors='coerce').fillna(0)

    team_attack = players_raw.groupby('team')[['expected_goals', 'expected_assists']].sum()
    teams['attack'] = team_attack['expected_goals'] + team_attack['expected_assists']
    gk_xgc = players_raw[players_raw['element_type'] == GOALKEEPER].groupby('team')['expected_goals_conceded'].sum()
    teams['defense'] = -gk_xgc.reindex(teams.index).fillna(gk_xgc.mean())
    venue_mean = teams[['strength_overall_home', 'strength_overall_away']].replace(0, np.nan).mean().mean()
    if pd.isna(venue_mean) or venue_mean == 0:
        venue_mean = 3.0
    teams['attack_home'] = teams['attack'] * (teams['strength_overall_home'].replace(0, venue_mean) / venue_mean)
    teams['attack_away'] = teams['attack'] * (teams['strength_overall_away'].replace(0, venue_mean) / venue_mean)
    teams['defence_home'] = teams['defense'] * (teams['strength_overall_home'].replace(0, venue_mean) / venue_mean)
    teams['defence_away'] = teams['defense'] * (teams['strength_overall_away'].replace(0, venue_mean) / venue_mean)

    return players_raw, teams, team_names


def build_players(players_raw, ppg_threshold=2, form_threshold=1):
    """
    Slim down + clean the raw players DataFrame into the fields used for rating.

    `ppg_threshold`/`form_threshold` filter out low-output players so squad-building
    and transfer-recommendation pools only consider rotation-proof/in-form players
    (the repo's original behaviour). Pass `None` for either to disable that filter -
    used to build the *full* player universe for name lookups/stat queries, so a
    player who has dropped out of form can still be looked up by the agent instead
    of being silently unresolvable.
    """
    players = players_raw[[
        "first_name", "second_name", "total_points", "points_per_game", "form", "now_cost",
        "selected_by_percent", "chance_of_playing_this_round", "element_type", "team", "id",
        "influence", "expected_goals_per_90", "expected_assists_per_90", "goals_scored", "assists",
        "expected_goals", "expected_assists", "status", "news", "minutes",
    ]].copy()
    players['name'] = players["first_name"] + " " + players["second_name"]
    players.set_index('name', inplace=True)

    players.drop(['first_name', 'second_name'], axis=1, inplace=True)
    players["ppg"] = pd.to_numeric(players["points_per_game"], errors='coerce').fillna(0)
    players.drop(['points_per_game'], axis=1, inplace=True)
    if ppg_threshold is not None:
        players = players[players["ppg"] > ppg_threshold]

    players["form"] = pd.to_numeric(players["form"], errors='coerce').fillna(0)
    if form_threshold is not None:
        players = players[players["form"] > form_threshold]
    players['count'] = 0
    players['expected_goals_per_90'] = pd.to_numeric(players['expected_goals_per_90'], errors='coerce').fillna(0)
    players['expected_assists_per_90'] = pd.to_numeric(players['expected_assists_per_90'], errors='coerce').fillna(0)
    players['goals_scored'] = pd.to_numeric(players['goals_scored'], errors='coerce').fillna(0)
    players['assists'] = pd.to_numeric(players['assists'], errors='coerce').fillna(0)
    players['selected_by_percent'] = pd.to_numeric(players['selected_by_percent'], errors='coerce').fillna(0)

    return players


def split_by_position(players):
    goalkeepers, defenders, midfielders, forwards = [], [], [], []
    for _, player in players.iterrows():
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
    return gk, df, md, fw


def _fixture_strengths(fixtures_df, teams):
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


def upcoming_fixtures(teams):
    """Fetch + enrich the list of unfinished fixtures with team attack/defense strength."""
    fixtures = requests.get(base_url + 'fixtures/').json()
    fixtures_df = pd.DataFrame(fixtures)
    fixtures_df = fixtures_df[fixtures_df['finished'] == False]
    fixtures_df = fixtures_df.sort_values(by=['event', 'kickoff_time'], na_position='last')
    return _fixture_strengths(fixtures_df, teams)


def team_upcoming_fixtures(team_id, fixtures_df, lookahead=FDR_LOOKAHEAD):
    """Return the next `lookahead` fixtures (raw rows) for a given team id."""
    return fixtures_df[
        (fixtures_df['team_h'] == team_id) | (fixtures_df['team_a'] == team_id)
    ].head(lookahead)


def add_fdr(player_df, fixtures_df, lookahead=FDR_LOOKAHEAD):
    """Attach a per-player fixture-difficulty-rating (fdr) column based on upcoming fixtures."""
    if player_df.empty:
        player_df = player_df.copy()
        player_df['fdr'] = pd.Series(dtype=float)
        return player_df

    scores = []
    for _, player in player_df.iterrows():
        team_id = player['team']
        team_fixtures = team_upcoming_fixtures(team_id, fixtures_df, lookahead)

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


def player_fixture_breakdown(player_row, fixtures_df, team_names, lookahead=3):
    """
    Per-fixture (not averaged) difficulty breakdown for a single player's next
    `lookahead` fixtures: gameweek, opponent, venue, and the same fdr formula
    used in add_fdr, so callers can show "next 3 fixtures" individually.
    """
    team_id = player_row['team']
    team_fixtures = team_upcoming_fixtures(team_id, fixtures_df, lookahead)

    breakdown = []
    for _, fixture in team_fixtures.iterrows():
        is_home = fixture['team_h'] == team_id
        own_att, own_def, opp_att, opp_def = _own_opp_strengths(fixture, is_home)
        attacking_fdr = own_att - opp_def
        defending_fdr = own_def - opp_att

        if player_row['element_type'] in (MIDFIELDER, FORWARD):
            fdr = attacking_fdr
        elif player_row['element_type'] == DEFENDER:
            weight = _defender_attack_weight(
                player_row.get('goals_scored', 0),
                player_row.get('assists', 0),
                player_row.get('expected_goals_per_90', 0),
                player_row.get('expected_assists_per_90', 0),
            )
            fdr = defending_fdr + weight * attacking_fdr
        else:
            fdr = defending_fdr

        opponent_id = fixture['team_a'] if is_home else fixture['team_h']
        breakdown.append({
            "gameweek": fixture.get('event'),
            "opponent": team_names[opponent_id] if opponent_id < len(team_names) else str(opponent_id),
            "venue": "H" if is_home else "A",
            "fdr": round(float(fdr), 3),
        })

    return breakdown


def rank_players(gk, df, md, fw, constraints=RANK_CONSTRAINTS):
    """Apply the existing multi-criteria rank-sum scoring (lower `count` = better) per position."""
    gk, df, md, fw = gk.copy(), df.copy(), md.copy(), fw.copy()
    for constraint in constraints:
        gk.sort_values(by=constraint, ascending=False, inplace=True)
        df.sort_values(by=constraint, ascending=False, inplace=True)
        md.sort_values(by=constraint, ascending=False, inplace=True)
        fw.sort_values(by=constraint, ascending=False, inplace=True)
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

    return gk, df, md, fw


def load_rated_players(ppg_threshold=2, form_threshold=1):
    """
    End-to-end load of the FPL player universe: fetch live data from the FPL API,
    enrich with team attack/defense strength, fixture difficulty (fdr), and the
    existing multi-criteria rank score (count). This is the single entry point
    the FPL agent tools rely on so there is exactly one rating pipeline.

    Returns a dict with:
      teams, team_names, fixtures, players (concatenated + ranked),
      by_position: {GOALKEEPER: gk, DEFENDER: df, MIDFIELDER: md, FORWARD: fw}
    """
    r = fetch_bootstrap()
    players_raw, teams, team_names = build_teams(r)
    players = build_players(players_raw, ppg_threshold=ppg_threshold, form_threshold=form_threshold)
    # Unfiltered universe (every player, regardless of form/ppg) so callers can still
    # look up out-of-form/returning-from-injury players by name instead of only the
    # curated "currently good" pool used for ranking/squad-building.
    players_all = build_players(players_raw, ppg_threshold=None, form_threshold=None)

    fixtures = upcoming_fixtures(teams)

    gk, df, md, fw = split_by_position(players)
    gk = add_fdr(gk, fixtures)
    df = add_fdr(df, fixtures)
    md = add_fdr(md, fixtures)
    fw = add_fdr(fw, fixtures)

    gk, df, md, fw = rank_players(gk, df, md, fw)

    all_players = pd.concat([gk, df, md, fw])
    all_players.sort_values(by=['count', 'ppg'], ascending=[True, False], inplace=True)

    return {
        "teams": teams,
        "team_names": team_names,
        "fixtures": fixtures,
        "players": all_players,
        "players_all": players_all,
        "by_position": {
            GOALKEEPER: gk,
            DEFENDER: df,
            MIDFIELDER: md,
            FORWARD: fw,
        },
        "players_raw": players_raw,
    }


def pick_squad(players, budget=1000, gk_size=2, def_size=5, mid_size=5, fwd_size=3, max_per_team=3):
    """Greedy squad builder using the ranked `players` DataFrame (existing algorithm, unchanged)."""
    positions = [0, 0, 0, 0, 0]
    iteams = [0] * 21
    team = []
    cost = 0
    i = 0
    c = 0

    while len(team) != 3:
        player = players.iloc[i]
        if (positions[player['element_type'] - 1] < [0, gk_size, def_size, mid_size, fwd_size][player['element_type']]
                and iteams[player['team']] < max_per_team):
            team.append([player.name, player['element_type'], player['id']])
            cost += player['now_cost']
            positions[player['element_type'] - 1] += 1
            iteams[player['team']] += 1
            c += 1
            i += 1

    budget -= cost
    s = 5

    while len(team) != 15:
        pavg = budget / (15 - c)
        player = players.iloc[i]
        if (positions[player['element_type'] - 1] < [0, gk_size, def_size, mid_size, fwd_size][player['element_type']]
                and iteams[player['team']] < max_per_team and pavg + s >= player['now_cost']
                and player.name not in [x[0] for x in team]):
            team.append([player.name, player['element_type'], player['id']])
            cost += player['now_cost']
            budget -= player['now_cost']
            positions[player['element_type'] - 1] += 1
            iteams[player['team']] += 1
            c += 1
            i = 2
            s *= -1

        i += 1

    team = sorted(team, key=lambda x: 1000 * x[1] + players.loc[x[0]]['count'])
    return team, cost, budget


def rebuild_squad(players, players_all, locked_names, excluded_names, budget,
                   gk_size=2, def_size=5, mid_size=5, fwd_size=3, max_per_team=3):
    """
    Build a 15-man squad made of `locked_names` (players being kept) plus the
    best-ranked affordable replacements for the remaining slots - reusing the
    exact same greedy, budget-balancing heuristic as `pick_squad` (the repo's
    existing "best team" algorithm), just seeded with a fixed core instead of
    building entirely from scratch.

    `players` is the ranked candidate pool (as returned by `load_rated_players`,
    already sorted by rank `count`) used to pick replacements.
    `players_all` is the unfiltered universe, used to resolve cost/team/position
    for the locked players (who may currently sit outside the ranked pool - e.g.
    injured/out of form - but are still being kept).
    `excluded_names` are never considered as replacements (typically the players
    being transferred out - without this they could be picked straight back).
    `budget` is the total 0.1m-unit budget available for the slots NOT locked
    (i.e. total squad budget minus the locked players' cost).

    Returns (team, cost, budget_remaining) exactly like `pick_squad`, where
    `team` is a list of [player_name, element_type, player_id].
    """
    limits = [0, gk_size, def_size, mid_size, fwd_size]
    positions = [0, 0, 0, 0, 0]
    iteams = [0] * 21
    team = []
    cost = 0

    for name in locked_names:
        row = players_all.loc[name]
        element_type = int(row['element_type'])
        team.append([name, element_type, int(row['id'])])
        cost += int(row['now_cost'])
        positions[element_type - 1] += 1
        iteams[int(row['team'])] += 1

    blocked = set(excluded_names) | set(locked_names)
    c = len(team)
    i = 0
    s = 5

    while len(team) != 15:
        if i >= len(players):
            raise RuntimeError(
                "Ran out of affordable/eligible candidates before filling all 15 slots "
                "- the remaining budget is likely too tight for the positions still needed."
            )
        pavg = budget / (15 - c)
        player = players.iloc[i]
        if (positions[player['element_type'] - 1] < limits[player['element_type']]
                and iteams[player['team']] < max_per_team and pavg + s >= player['now_cost']
                and player.name not in blocked):
            team.append([player.name, player['element_type'], player['id']])
            cost += player['now_cost']
            budget -= player['now_cost']
            positions[player['element_type'] - 1] += 1
            iteams[player['team']] += 1
            blocked.add(player.name)
            c += 1
            i = 2
            s *= -1
        i += 1

    def _sort_key(entry):
        name, element_type, _id = entry
        count = players.loc[name]['count'] if name in players.index else 0
        return 1000 * element_type + count

    team = sorted(team, key=_sort_key)
    return team, cost, budget


def print_team(team, players, team_names, cost, budget):
    print("----------------------------------")
    print("Team:")
    for i in team:
        print(i[0], players.loc[i[0]]['count'])
        print(players.loc[i[0]]['total_points'])
        print(players.loc[i[0]]['now_cost'] / 10)
        print(team_names[(players.loc[i[0]]['team'])])

    print("Total Cost:", cost)
    print("Remaining Budget:", budget)


def tp_352(team, players, team_names):
    tp = 0
    s = ""
    print(("3-5-2:").center(100))
    print("\n")
    for i in [0, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13]:
        name = team[i][0].split()
        s += name[0][0] + ". " + name[-1] + "  " "(" + team_names[(players.loc[team[i][0]]['team'])] + ")   "
        if (i in [0, 4, 11, 13]):
            print(s.center(100), "\n")
            s = ''
        tp += players.loc[team[i][0]]['total_points']
    tp += players.iloc[0]['total_points']
    print(("Total Points: " + (str(tp))).center(100))
    print("\n")
    return tp


def tp_442(team, players, team_names):
    tp = 0
    s = ""
    print(("4-4-2:").center(100))
    print("\n")
    for i in [0, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13]:
        name = team[i][0].split()
        s += name[0][0] + ". " + name[-1] + "  " "(" + team_names[(players.loc[team[i][0]]['team'])] + ")"
        if (i in [0, 5, 10, 13]):
            print(s.center(100), "\n")
            s = ''
        tp += players.loc[team[i][0]]['total_points']
    tp += players.iloc[0]['total_points']
    print(("Total Points: " + (str(tp))).center(100))
    print("\n")
    return tp


def tp_343(team, players, team_names):
    tp = 0
    s = ""
    print(("3-4-3:").center(100))
    print("\n")
    for i in [0, 2, 3, 4, 7, 8, 9, 10, 12, 13, 14]:
        name = team[i][0].split()
        s += name[0][0] + ". " + name[-1] + "  " + "(" + team_names[(players.loc[team[i][0]]['team'])] + ")"
        if (i in [0, 4, 10, 14]):
            print(s.center(100), "\n")
            s = ''
        tp += players.loc[team[i][0]]['total_points']
    tp += players.iloc[0]['total_points']
    print(("Total Points: " + (str(tp))).center(100))
    print("\n")
    return tp


def current_event_id(bootstrap):
    """
    Determine the gameweek whose squad reflects a manager's *current* team.
    Prefers the event flagged 'is_current' (true right up until the next
    deadline passes, even after that gameweek's matches finish) since that's
    when the manager's picks still represent their live squad. Falls back to
    the next upcoming event, then the most recently finished one.
    """
    events = bootstrap.get('events', [])
    current = next((e for e in events if e.get('is_current')), None)
    if current is None:
        current = next((e for e in events if e.get('is_next')), None)
    if current is None:
        finished = [e for e in events if e.get('finished')]
        current = finished[-1] if finished else (events[0] if events else None)
    return current['id'] if current else None


def fetch_manager_entry(manager_id):
    """Fetch a manager's public entry info (team name, manager name, bank/value)."""
    resp = requests.get(f'{base_url}entry/{manager_id}/')
    if resp.status_code == 404:
        raise ValueError(f"No FPL manager found with ID {manager_id}.")
    resp.raise_for_status()
    return resp.json()


def fetch_manager_picks(manager_id, event_id):
    """Fetch a manager's 15-player squad (picks) for a given gameweek."""
    resp = requests.get(f'{base_url}entry/{manager_id}/event/{event_id}/picks/')
    if resp.status_code == 404:
        raise ValueError(
            f"No picks found for manager {manager_id} in gameweek {event_id} "
            "(the team may not have been set for this gameweek yet)."
        )
    resp.raise_for_status()
    return resp.json()


def load_manager_squad(manager_id, bootstrap=None):
    """
    Fetch a manager's live current squad from the FPL API and return it as a
    plain list of player dicts (one per squad member), enriched with each
    player's name/team/price from the same bootstrap data used everywhere
    else in this pipeline - no separate/duplicate data source.

    Returns a dict with manager/team metadata plus `squad`: a list of 15
    player dicts ordered by squad slot (1-11 starting XI, 12-15 bench).
    """
    bootstrap = bootstrap or fetch_bootstrap()
    players_raw, teams, team_names = build_teams(bootstrap)
    players_all = build_players(players_raw, ppg_threshold=None, form_threshold=None)
    players_by_id = players_all.reset_index().set_index('id')

    event_id = current_event_id(bootstrap)
    if event_id is None:
        raise ValueError("Could not determine the current FPL gameweek from the bootstrap data.")

    entry = fetch_manager_entry(manager_id)
    picks_payload = fetch_manager_picks(manager_id, event_id)
    picks = picks_payload.get('picks', [])
    history = picks_payload.get('entry_history', {})

    squad = []
    for pick in picks:
        element_id = pick['element']
        if element_id not in players_by_id.index:
            continue
        row = players_by_id.loc[element_id]
        squad.append({
            "player_name": row['name'],
            "element_type": int(row['element_type']),
            "team": team_names[row['team']] if row['team'] < len(team_names) else str(row['team']),
            "price_m": round(row['now_cost'] / 10, 1),
            "squad_slot": pick['position'],
            "on_bench": pick['position'] > 11,
            "is_captain": bool(pick.get('is_captain')),
            "is_vice_captain": bool(pick.get('is_vice_captain')),
            "multiplier": pick.get('multiplier', 1),
        })
    squad.sort(key=lambda p: p["squad_slot"])

    return {
        "manager_id": int(manager_id),
        "team_name": entry.get('name'),
        "manager_name": f"{entry.get('player_first_name', '')} {entry.get('player_last_name', '')}".strip(),
        "gameweek": event_id,
        "bank_m": round(history.get('bank', 0) / 10, 1),
        "team_value_m": round(history.get('value', 0) / 10, 1),
        "squad": squad,
    }


def get_gameweek_history(player_id):
    r = requests.get(
        base_url + 'element-summary/' + str(player_id) + '/'
    ).json()
    df = pd.json_normalize(r['history'])
    return df


def main():
    data = load_rated_players()
    players = data["players"]
    by_position = data["by_position"]
    team_names = data["team_names"]

    gk, df, md, fw = by_position[GOALKEEPER], by_position[DEFENDER], by_position[MIDFIELDER], by_position[FORWARD]

    print(gk.head(7)[['ppg', 'form', 'fdr', 'count']])
    print(df.head(7)[['ppg', 'form', 'fdr', 'count']])
    print(md.head(7)[['ppg', 'form', 'fdr', 'count']])
    print(fw.head(7)[['ppg', 'form', 'fdr', 'count']])

    team, cost, budget = pick_squad(players)
    print_team(team, players, team_names, cost, budget)
    tp_352(team, players, team_names)


if __name__ == "__main__":
    main()

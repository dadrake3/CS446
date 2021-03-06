SELECT
  team.school_name as team_name,
  opponent.school_name as opponent_name,
  team.game_id AS game_id,
  team.season AS season,
  team.scheduled_date AS scheduled_date,
  team.team_id AS team_id,
  team.minutes AS minutes,
  team.is_win AS win,
  opponent.team_id AS opponent_id,
  team.home_team AS home,
  team.points AS team_points,
  opponent.points AS opponent_points,
  team.fast_break_pts AS team_fast_break_pts,
  team.second_chance_pts AS team_second_chance_pts,
  team.field_goals_made AS team_field_goals_made,
  team.field_goals_att AS team_field_goals_att,
  team.field_goals_pct AS team_field_goals_pct,
  team.three_points_made AS team_three_points_made,
  team.three_points_att AS team_three_points_att,
  team.three_points_pct AS team_three_points_pct,
  team.two_points_made AS team_two_points_made,
  team.two_points_att AS team_two_points_att,
  team.two_points_pct AS team_two_points_pct,
  team.free_throws_made AS team_free_throws_made,
  team.free_throws_att AS team_free_throws_att,
  team.free_throws_pct AS team_free_throws_pct,
  team.ts_pct AS team_ts_pct,
  team.efg_pct AS team_efg_pct,
  team.rebounds AS team_rebounds,
  team.offensive_rebounds AS team_offensive_rebounds,
  team.defensive_rebounds AS team_defensive_rebounds,
  team.dreb_pct AS team_dreb_pct,
  team.oreb_pct AS team_oreb_pct,
  team.steals AS team_steals,
  team.blocks AS team_blocks,
  team.blocked_att AS team_blocked_att,
  team.assists AS team_assists,
  team.turnovers AS team_turnovers,
  team.team_turnovers AS team_team_turnovers,
  team.points_off_turnovers AS team_points_off_turnovers,
  team.assists_turnover_ratio AS team_assists_turnover_ratio,
  team.ast_fgm_pct AS team_ast_fgm_pct,
  team.personal_fouls AS team_personal_fouls,
  team.flagrant_fouls AS team_flagrant_fouls,
  team.player_tech_fouls AS team_player_tech_fouls,
  team.team_tech_fouls AS team_team_tech_fouls,
  team.coach_tech_fouls AS team_coach_tech_fouls,
  team.ejections AS team_ejections,
  team.foulouts AS team_foulouts,
  team.score_delta AS team_score_delta,
  team.opp_score_delta AS team_opp_score_delta,
  team.possessions AS team_possession,
  opponent.fast_break_pts AS opponent_fast_break_pts,
  opponent.second_chance_pts AS opponent_second_chance_pts,
  opponent.field_goals_made AS opponent_field_goals_made,
  opponent.field_goals_att AS opponent_field_goals_att,
  opponent.field_goals_pct AS opponent_field_goals_pct,
  opponent.three_points_made AS opponent_three_points_made,
  opponent.three_points_att AS opponent_three_points_att,
  opponent.three_points_pct AS opponent_three_points_pct,
  opponent.two_points_made AS opponent_two_points_made,
  opponent.two_points_att AS opponent_two_points_att,
  opponent.two_points_pct AS opponent_two_points_pct,
  opponent.free_throws_made AS opponent_free_throws_made,
  opponent.free_throws_att AS opponent_free_throws_att,
  opponent.free_throws_pct AS opponent_free_throws_pct,
  opponent.ts_pct AS opponent_ts_pct,
  opponent.efg_pct AS opponent_efg_pct,
  opponent.rebounds AS opponent_rebounds,
  opponent.offensive_rebounds AS opponent_offensive_rebounds,
  opponent.defensive_rebounds AS opponent_defensive_rebounds,
  opponent.dreb_pct AS opponent_dreb_pct,
  opponent.oreb_pct AS opponent_oreb_pct,
  opponent.steals AS opponent_steals,
  opponent.blocks AS opponent_blocks,
  opponent.blocked_att AS opponent_blocked_att,
  opponent.assists AS opponent_assists,
  opponent.turnovers AS opponent_turnovers,
  opponent.points_off_turnovers AS opponent_points_off_turnovers,
  opponent.assists_turnover_ratio AS opponent_assists_turnover_ratio,
  opponent.ast_fgm_pct AS opponent_ast_fgm_pct,
  opponent.personal_fouls AS opponent_personal_fouls,
  opponent.flagrant_fouls AS opponent_flagrant_fouls,
  opponent.player_tech_fouls AS opponent_player_tech_fouls,
  opponent.team_tech_fouls AS opponent_tech_fouls,
  opponent.coach_tech_fouls AS opponent_coach_tech_fouls,
  opponent.ejections AS opponent_ejections,
  opponent.foulouts AS opponent_foulouts,
  opponent.score_delta AS opponent_score_delta,
  opponent.opp_score_delta AS opponent_opp_score_delta,
  opponent.possessions AS opponent_possession

FROM
  `NCAA_mbb1.team_games_individual_no_cume` AS team
JOIN
   `NCAA_mbb1.team_games_individual_no_cume` AS opponent
ON
  team.game_id = opponent.game_id AND team.team_id != opponent.team_id
-- WHERE
--   team.home_team = true 
ORDER BY team_name, scheduled_date


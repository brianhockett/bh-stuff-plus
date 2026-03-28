# DS 4320 Project 1: MLB Stuff+ Database

### Executive Summary

Executive summary paragraph


| Information | Value  |
|---:|---:|
| Name | Brian Hockett |
| NetID | mgh2xx |
| DOI | [https://doi.org/TBD](https://doi.org/TBD) |
| Press Release | [New Stuff+ Model Shows Predictive Advantage Over Traditional Performance Metrics](https://github.com/brianhockett/bh-stuff-plus/blob/main/PressRelease.md) |
| Data | [UVA OneDrive Folder Link](TBD) |
| Pipeline | [Analysis Pipeline](TBD) |
| License | [TBD](TBD.md) |

## Problem Definition

### General and Specific Problem
- **General Problem:** Projecting athletic performance. Projecting the performance of MLB pitchers.
- **Specific Problem:** Using MLB Statcast Data for the 2021-2025 Seasons and Expected Run Values, design a machine learning model to predict the performance of MLB Pitchers based on the quality of their individual pitches in terms of their expected run value, standardized into a normally distributed `Stuff+` value, validating with next-season statistical outcomes.

### Rationale
The first step in the refinement was defining the definition of "performance", a very broad term for athletes. The choice was made to use expected run value as a measure of performance, as it removes the effects of defense, ballpark, and luck, allowing for performance attribution to be given solely to the pitcher. Likewise, by focusing solely on pitch characteristics for the refined problem, effects outside of the pitcher's control are ignored, allowing for the performance projection to be tied directly to actions in the pitcher's control. Deciding to normalize the model output into a `Stuff+` metric was done to allow for easier interpretation of results by athletes, coaches, and fans. The 2021-2025 time range was defined in order to capture a sufficient sample size for the model, as well as to allow for validation of results with future outcomes, to ensure the solution is indeed addressing this refined problem.

### Motivation
Traditionally, Major League Baseball teams and fans have relied on outcome-based metrics like ERA, Wins, Strikeouts, etc. to evaluate pitcher performance. Despite this, these metrics are not actually predictive of future success, nor do they provide information about how a player could improve their future performance beyond, "allow less runs", which is not useful in a player development context. As such, metrics for evaluating pitcher performance should have the following qualities: predictive of future success and actionable paths for improvement. This is where the motivation for the creation of `Stuff+` models comes from. The outputs of `Stuff+` models allow for prediction of a pitcher's future success, even if their current results are subpar. Likewise, because machine learning is used to create `Stuff+` models, the contribution of individual pitch characteristics to the `Stuff+` value allows for more concentrated coaching and paths for improvement, such as "Change the grip on your slider to add 2 inches of Horizontal Break." Overall, this project is motivated by the goal of quantifying pitcher performance in a way that is both predictive of future success and actionable for player development.

### Press Release Headline and Link
[**New Stuff+ Model Shows Predictive Advantage Over Traditional Performance Metrics**](https://github.com/brianhockett/bh-stuff-plus/blob/main/PressRelease.md)

## Domain Exposition

### Terminology
**Statcast & Stuff+ Terminology**

| Term | Abbr | Definition | Unit | Typical Range | Better? |
|------|------|-----------|------|-------|---------|
| Velocity | - | Ball speed at release | mph | 75-105 | Higher |
| Spin Rate | - | Ball rotational speed at release | rpm | 1200-3000 | N/A |
| Extension | - | Arm extension towards plate at release | ft | 5.5-7.5 | N/A |
| Induced Vertical Break | IVB | Vertical deviation above gravity-only path | inches | -20 to +20 | Pitch-dependent |
| Horizontal Break | HB | Lateral movement after release | inches | -20 to +20 | Pitch-dependent |
| Spin Axis | SA | Direction of ball spin/rotation | degrees | 0-360 | Pitch-dependent |
| Expected Run Value | xRV | Runs expected to score based on pitch outcome | runs | -1 to +2 | Lower |
| Innings Pitched | IP | Total innings pitched in season | innings | 30.0-220.0 | N/A |
| Strikeouts per 9 IP | K/9 | Strikeout rate normalized to 9 innings | ratio | 6.0-11.0+ | Higher |
| Walks per 9 IP | BB/9 | Walk rate normalized to 9 innings | ratio | 1.0-5.0 | Lower |
| Earned Run Average | ERA | Earned runs per 9 innings pitched | ratio | 2.00-5.50 | Lower |
| Fielding Independent Pitching | FIP | Pitcher performance independent of defense | ratio | 2.5-4.5 | Lower |
| Whiff Rate | Whiff% | % of swings that miss the pitch | percent | 10%-40% | Higher |
| Stuff+ Score | Stuff+ | Predicted pitch quality metric (mean 100, standardized) | scale | 80-120 | Higher

### Background Information
This project lives in the Sabermetrics domain of Major League Baseball. Sabermetrics refers to the empirical, analytical approach to understanding the sport of baseball, and the performance of baseball players. This specific project uses Sabermetrics to analyze the performance of individual pitches, and understand the characteristics of pitches that lead to positive outcomes. Since 2015, Major League Baseball has used Statcast, a high-speed camera tracking system, to capture the ball movement, swings, and batted-ball results for every pitch thrown at the major league level. This granular pitch-level data is used for a number of purposes in the Sabermetrics domain, from analysis of how pitch velocity and movement factors into results, to modeling batter swing-paths in order to increase exit velocity. Through this precise, objective data, sabermetrics enables a deeper understanding of the underlying mechanics of pitcher performance, moving beyond traditional outcome-based statistics towards predictive, process-based evaluation.

### Background Reading
| Title | Description | Link |
|-------|-------------|------|
| Statcast at 10: From MLB’s secret project to inescapable part of modern baseball | NYT The Athletic article covering Statcast's introduction, growth, and impact on baseball analytics and player evaluation | ENTER ONEDRIVE LINK |
| What is Stuff+ and How Can it Help You? | Rockland Peak Performance overview of Stuff metrics, explanation of Stuff+ calculation, and application for pitcher development and performance analysis | ENTER ONEDRIVE LINK |
| Pitch Design: What is Stuff+? Quantifying Pitches with Pitch Models | Driveline's technical guide to their version of the Stuff+ metric, and how they implement it for player development | ENTER ONEDRIVE LINK |
| What the latest Statcast upgrade makes possible | MLB.com article covering the 2020 improvement of Statcast systems, which enables pose tracking technology for more detailed analysis | ENTER ONEDRIVE LINK |
| Baseball Spin Rate Basics and Pitch Movement | Rockland Peak Performance guide to understanding spin rate and pitch movement | ENTER ONEDRIVE LINK |

## Data Creation

### Data Acquisition
The raw data was acquired in four steps, one for each table in the database. The primary table, StatcastPitch, was acquired first, using the python package `pybaseball`. The `statcast` function of `pybaseball` was used to collect the Statcast data for every pitch thrown in the regular season from the start of the 2021 season to the end of the 2025 season. The choice was made to limit the number of fields from the Statcast data that would be retained, in order to limit the scope of the dataset to that required for this project. The retained fields can be found in the Data Dictionary. Next, the PitcherBio table data was acquired via the `MLB Stats API`. An API call was sent for every unique pitcher in the StatcastPitch dataset, in order to ensure every pitcher would be present in both tables. The responses of the API calls were collectively stored to become the PitcherBio table. The next step was to acquire the PitcherStats table data, which was done through the `Fangraphs API`. For every season from 2021 to 2025, an API call was sent, which returned all pitchers' season summary statistics for each season. This data was stored to become the PitcherStats table. Lastly, the ExpectedRunValue table was derived via the Statcast data. The average change in run expectancy for each balls-strikes-outcome combination in the Statcast data was calculated and stored to become the ExpectedRunValue table.

Each dataset was saved as both a .csv and .parquet file, and will be loaded into a DuckDB database for the project submission.

### Code Table
| Data | Description | Link to Code |
|-------------|-----------|-------------|
| Statcast Pitch Data | Uses `pybaseball` package to get Statcast pitch data from 2021 to 2025 | https://github.com/brianhockett/bh-stuff-plus/blob/main//src/generate-statcast.py |
| Pitcher Statistics Data | Uses `Fangraphs API` to collect pitcher summary statistics from 2021 to 2025 | https://github.com/brianhockett/bh-stuff-plus/blob/main//src/generate-stats.py |
| Pitcher Biographical Data | Uses `MLB Stats API` to collect pitcher biographical information | https://github.com/brianhockett/bh-stuff-plus/blob/main/src/generate-bio.py |
| Expected Run Value Data | Uses Statcast pitch data to derive average Expected Run Value deltas for every balls-strikes-outcome combination | https://github.com/brianhockett/bh-stuff-plus/blob/main/src/generate-run-values.py |

### Bias Identification
# NEEDS FIXING
Bias may have been introduced in a number of ways during the data collection process. Survivorship bias may be the most prominent form of bias in the collection of the Statcast dataset, as pitcher injuries are very common in MLB. Injured pitchers will make up a significantly lower proportion of the dataset than they would if they were healthy, and may have notably different pitch characteristics than healthy pitchers. For example, pitchers with high velocity get injured more frequently, and will therefore make up less of the dataset than they otherwise would. Secondly, the very small sample size of some combinations in the ExpectedRunValue table may skew the results of the analysis later on. Certain combinations, such as triples on a 3-0 count, occur very rarely in MLB games, which could lead to outlier average run values, where a small number of high or low value instances of this combination cause the average to drastically over- or under- estimate the true value of that outcome. Lastly, measurement bias inherent to the Hawk-Eye tracking system may bias the results of pitch tracking from stadium-to-stadium. While every MLB stadium has the same number of Hawk-Eye cameras, it is impossible to configure them identically due to the non-uniformity of MLB ballparks. This could lead to systematic differences in the measurements of different stadiums.

### Bias Mitigation
# MAY NEED FIXING
Several steps were and can be taken during data collection and analysis processes to mitigate potential biases. One of these steps was filtering the Statcast data to include only regular season games via the `game_type` field. This ensured that spring training and playoff data were excluded from the dataset, as they would skew the pitch characteristics, with spring training pitches being of lower quality, and playoff pitches of higher quality. Anomalous and ultra-rare observations identified during data validation, such as walks occurring in impossible counts and hit-by-pitch records with no corresponding event, were also removed. This ensured that the ExpectedRunValue table would be as unbiased as possible when building the `Stuff+` model. In the analysis phase, handedness mirroring of horizontal features will be applied to normalize differences between left- and right- handed pitchers, to ensure these features do not work against themselves in the model.

Unfortunately, not all potential biases can be fully mitigated. Survivorship bias from pitcher injuries is an inherent limitation of any MLB pitcher dataset. Similarly, stadium-level measurement bias from Hawk-Eye configuration differences cannot be corrected within the Statcast dataset. These biases should be acknowledged as limitations of the dataset and considered when interpreting the analysis results.

### Rationale
The first major decision made was about which seasons of data to collect. The choice to collect the 2021-2025 seasons was made, because the Hawk-Eye system was introduced in 2020, meaning all 5 seasons in the Statcast dataset will have been collected using the same system. The 2020 season was excluded, due to the smaller sample size (shorter season) and overall non-standard playing environment of that season due to the Covid-19 pandemic. Another choice that had to be made in the Statcast data was how to handle the catcher interference, fielding error, truncated plate appearance, automatic ball, and automatic strike events. While these events have run expectancies associated with them, the pitcher lacks any agency in these 5 outcomes. The choice was made to null their contributions to the run expectancy table, as the purpose of the project is to analyze and predict controllable pitcher performance. The last major judgement call in the data collection process that had to be made was what version of delta run expectancy to use. There were two options: one which took into account whether there were runners on each of the 3 bases, and one which ignored the base states by averaging over all outcomes regardless of base state. The choice was made to ignore the base state, in order to ensure the quality of the pitch is agnostic to the situation the pitcher is in, which is often very noisy. By ignoring base states, relief pitchers who come into the game with runners on and pitchers who allow runners to reach base for reasons out of their control are not under- or over- valued by the run expectancy table. This will produce a less noisy, and ideally more predictive result.

## Metadata

### Schema
![Entity Relationship Diagram](https://github.com/brianhockett/bh-stuff-plus/blob/main/img/erd.png)

### Data Table
| Table | Description | Size (CSV) | Size (Parquet) | Link |
|-------------|-----------|---|---|-------------|
| StatcastPitch | Outcome and pitch characteristics for every pitch captured by Statcast from 2021 to 2025| 812 MB | 160 MB| [StatcastPitch Data](https://myuva-my.sharepoint.com/:u:/g/personal/mgh2xx_virginia_edu/IQCFA8qjA3E_RqE-ENFCZfdUAZCmkKZUyp-EI5O5hFdWetY?e=rjPyh5) |
| PitcherStats | Summary statistics for every pitcher-season from 2021 to 2025| 573 KB |258 KB | [PitcherStats Data](https://myuva-my.sharepoint.com/:u:/g/personal/mgh2xx_virginia_edu/IQDajyVuscayQ6UCZXz_7kuzAf_a99sb0C5brWAD1rdgA2E?e=Bpgf8V) |
| PitcherBio | Biographical information for every pitcher found in the StatcastPitch table |67 KB |40 KB | [PitcherBio Data](https://myuva-my.sharepoint.com/:u:/g/personal/mgh2xx_virginia_edu/IQDloHJW6zg-SZRE-kwDATInASCnpoVCrockz90cpFScRd4?e=qcQMGJ)|
| ExpectedRunValue | Expected Run Value delta for every balls-strikes-outcome combination |6 KB |6 KB | [ExpectedRunValue Data](https://myuva-my.sharepoint.com/:u:/g/personal/mgh2xx_virginia_edu/IQDjTWGguuSeRLo_655pzODlAZzjtuIHCEeQYJqDTPipGCM?e=k8aZuF) |

### Data Dictionary
#### **StatcastPitch**
| Field Name | Data Type | Description | Example Value | Key |
|-------------|-----------|-------------|---------------|-----|
| pitchID | Integer | Unique identifier for each pitch in the dataset, coutning up from 0 | 1349 | Primary |
| pitcher | Integer | MLBAM identifer for each pitcher in the dataset | 445276 | Foreign |
| player_name | String | Name of the pitcher | "Alcala, Jorge" | |
| p_throws | String | Handedness of the pitcher | "L" | |
| game_date | Datetime | Date of the game pitch was thrown in | "2024-05-13" |
| game_year | Integer | Year of the game pitch was thrown in | 2021 |
| pitcher_season | String | Composite foreign key combining pitcher and game_year columns | "445276_2021" | Foreign |
| balls | Integer | The number of balls in the count prior to the pitch | 3 |  |
| strikes | Integer | The number of strikes in the count prior to the pitch | 2 |  |
| pitch_type | String | The short-form name for the type of pitch thrown | "FF" |  |
| pitch_name | String | The long-form name for the type of pitch thrown | "4-Seam Fastball" |  |
| release_speed | Float | The speed of the ball at the time of release in mph | 94.8 |  |
| release_pos_x | Float | Horizontal release position of the ball measured in feet from the catcher's perspective | -2.62 |  |
| release_pos_y | Float | Release position of the ball measured in feet from the catcher's perspective | 54.63 |  |
| release_pos_z | Float | Vertical release position of the ball measured in feet from the catcher's perspective | 6.42 |  |
| release_extension | Float | Release extension of pitch in feet towards the plate | 5.9 |  |
| release_spin_rate | Integer | Spin rate in rpm of pitch at release | 2400 |  |
| spin_axis | Float | Spin axis in the 2D X-Z plane in degrees from 0 to 360 | 225 |  |
| pfx_x | Float | Horizontal movement in inches from the catcher's perspective| -9.1 |  |
| pfx_z | Float | Vertical deviation from gravity-only path in inches from the catcher's perspective| 19.1 |  |
| ax | Float | Horizontal acceleration of the ball at y = 50ft in ft/sec$^2$| -11.74 |  |
| ay | Float | Acceleration of the ball towards plate at y = 50ft in ft/sec$^2$| 33.52 |  |
| az | Float | Vertical acceleration of the ball at y = 50ft in ft/sec$^2$| -12.75 |  |
| description | String | Description of the resulting pitch| "missed_bunt" |  |
| description_group | String | Broader, grouped description of the resulting pitch | "swinging_strike" |  |
| events | String | Outcome of the resulting plate appearance, if it ends on this pitch | "fielders_choice" |  |
| events_group | String | Broader, grouped outcome of the resulting plate appearance, if it ends on this pitch | "field_out" |  |
| delta_run_exp | Float | Difference in expected runs before and after pitch | 1.42 | |
| event_balls_strikes | String | Composite foreign key combining events_group/description_group, balls, and strikes | "home_run_0_2" | Foreign |

#### **PitcherBio**
| Field Name | Data Type | Description | Example Value | Key |
|-------------|-----------|-------------|---------------|-----|
| pitcherID | Integer | MLBAM identifier for each pitcher in the StatcastPitch dataset | 445276 | Primary |
| name | String | Name of the pitcher | "Jorge Alcala" |  |
| throws | String | Handedness of the pitcher | "R" | |
| age | Integer | Age of the pitcher at the time of the API call | 28 | |
| height | String | Height of the pitcher measured in feet and inches | "6' 3"" | |
| weight | Integer | Weight of the pitcher measured in lbs | 205 | |

#### **ExpectedRunValue**
| Field Name | Data Type | Description | Example Value | Key |
|-------------|-----------|-------------|---------------|-----|
| event_balls_strikes | String | Composite primary key combining events_group/description_group, balls, and strikes | "home_run_3_2" | Primary |
| event | String | Pitch/PA outcome | "home_run" | |
| balls | Integer | Number of balls in the count | 3 | |
| strikes | Integer | Number of strikes in the count | 2 | |
| delta_run_exp | Float | Change in expected runs based on pitch/PA outcome and count| 1.42 | |

#### **PitcherStats**
| Field Name | Data Type | Description | Example Value | Key |
|-------------|-----------|-------------|---------------|-----|
| pitcherID_Season | String | Composite primary key combining pitcher MLBAM identifer and season year columns | "445276_2021" | Primary |
| pitcherID | Integer | MLBAM identifer for each pitcher in the dataset | 445276 | Foreign |
| playerid| Integer | Fangraphs identifier for each pitcher in the dataset | 19361 | |
| Season | Integer | Year of the season | 2023 | |
| PlayerName | String | Name of the pitcher | "Corbin Burnes" | |
| Age | Float | Age of the pitcher at the start of the season | 26.0 | |
| Throws | String | Handedness of the pitcher | "R" | |
| ERA | Float | Earned Runs Allowed by pitcher per 9 innings pitched | 3.28 | |
| FIP | Float | Fielding-Independent Pitching; ERA-scale metric measuring pitcher performance separate from defensive impact | 3.61 | |
| xFIP | Float | Expected Fielding-Independent Pitching; FIP, accounting for randomness and park factor | 3.61 | |
| IP | Float | Number of innings pitched, where the decimal value refers to the number of outs recorded out of 3 | 182.1 | |
| Pitches | Integer | Number of pitches thrown in the season | 1321 | |
| WAR | Float | Wins Above Replacement; Measuring cumulative value added to team | 5.6 | |


### Data Dictionary Uncertainty
# NEEDS FIXING, COMBINE WITH ABOVE TABLE
| Field Name | Data Type | Reason for Uncertainty | Quantification of Uncertainty |
|-------------|-----------|-------------------------------|------|
| release_speed | Float | Systematic measurement error from the Hawk-Eye tracking system  | $\pm$0.1-0.3 mph |
| release_pos_x | Float | Systematic measurement error from the Hawk-Eye tracking system | $\pm$0.02 ft |
| release_pos_y | Float | Systematic measurement error from the Hawk-Eye tracking system | $\pm$0.02 ft |
| release_pos_z | Float | Systematic measurement error from the Hawk-Eye tracking system | $\pm$0.02 ft |
| release_extension | Float | Systematic measurement error from the Hawk-Eye tracking system | $\pm$0.02 ft |
| pfx_x | Float | Compounded errors from positional tracking when fitting trajectory model | $\pm$1 in  |
| pfx_z | Float | Compounded errors from positional tracking when fitting trajectory model | $\pm$1 in  |
| ax | Float | Compounded errors from positional tracking when fitting trajectory model | $\pm$2 ft/sec$^2$ |
| ay | Float | Compounded errors from positional tracking when fitting trajectory model | $\pm$2 ft/sec$^2$ |
| az | Float | Compounded errors from positional tracking when fitting trajectory model | $\pm$2 ft/sec$^2$ |
| release_spin_rate | Integer | Systematic measurement error from the Hawk-Eye tracking system | $\pm$25-50 rpm |
| spin_axis | Integer | Systematic measurement error from the Hawk-Eye tracking system. Higher for pitches with significant gyrospin | $\pm$2-10 deg |
| delta_run_exp | Float | Uncertainty determined by sample size, with small sample count-outcome combinations having greater uncertainty | $\pm$0.001-0.07 runs |



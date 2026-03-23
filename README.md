# DS 4320 Project 1: MLB Stuff+ Database

### Executive Summary

Executive summary paragraph


| Project Information |   |
|---:|---:|
| Name | Brian Hockett |
| NetID | mgh2xx |
| DOI | [https://doi.org/TBD](https://doi.org/TBD) |
| Press Release | [New Stuff+ Model Shows Predictive Advantage Over Traditional Performance Metrics](https://github.com/brianhockett/bh-stuff-plus/blob/main/PressRelease.md) |
| Data | [UVA OneDrive Folder Link](TBD) |
| Pipeline | [Analysis Code](TBD) |
| License | [TBD](TBD.md) |

## Problem Definition

### General and Specific Problem
- **General Problem:** General Problem Here
- **Specific Problem:** Specific Problem Here

### Rationale
Rationale paragraph

### Motivation
Motivation paragraph

### Press Release Headline and Link
[**New Stuff+ Model Shows Predictive Advantage Over Traditional Performance Metrics**](https://github.com/brianhockett/bh-stuff-plus/blob/main/PressRelease.md)

## Domain Exposition

### Terminology
Terminology Table

### Background Information
Domain explanation paragraph

### Background Reading
Table with links to readings (stored in UVA OneDrive Folder)

## Data Creation

### Data Acquisition
Data Acquisition paragraph

### Code Table
| | Data | Description | Link to Code |
|-|-------------|-----------|-------------|
| | Statcast Pitch Data | Uses `pybaseball` package to get Statcast pitch data from 2021 to 2025 | https://github.com/brianhockett/bh-stuff-plus/blob/main//src/generate-statcast.py |
| | Pitcher Statistics Data | Uses `Fangraphs API` to collect pitcher summary statistics from 2021 to 2025 | https://github.com/brianhockett/bh-stuff-plus/blob/main//src/generate-stats.py |
| | Pitcher Biographical Data | Uses `MLB Stats API` to collect pitcher biographical information | https://github.com/brianhockett/bh-stuff-plus/blob/main/src/generate-bio.py |
| | Expected Run Value Data | Uses Statcast pitch data to derive average Expected Run Value deltas for every balls-strikes-outcome combination | https://github.com/brianhockett/bh-stuff-plus/blob/main/src/generate-run-values.py |

### Bias Identification
Bias ID paragraph

### Bias Mitigation
Bias mitigation paragraph

### Rationale
Rationale paragraph

## Metadata

### Schema
ERD Image

### Data Table

### Data Dictionary

### Data Dictionary Uncertainty


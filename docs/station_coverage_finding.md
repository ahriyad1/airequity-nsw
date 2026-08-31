# Station coverage: registry vs operational network

The NSW Air Quality API `get_SiteDetails` endpoint returns 137 entries.
This is a site registry, not an inventory of operational monitors.

| Filter | Count |
|---|---|
| All API entries | 137 |
| Excluding region aggregates, test sites, non-Sydney | 24 |
| Reporting PM2.5 during 2023–2024 | 19 |

Five Sydney stations appear in the registry but returned no PM2.5 data:
Lindfield and Chullora (Sydney East), Bargo and Macarthur (South-west),
Vineyard (North-west). Testing PM2.5 availability at these sites across
2018, 2020, 2022, 2025 and 2026 returned no records in any year.

Implications:
- The operational Sydney network is 19 PM2.5 stations, not 24 or 137
- Leave-one-station-out validation runs across 19 folds
- Reported network size should not be taken from the registry endpoint

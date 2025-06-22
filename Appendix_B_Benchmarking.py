df['Round'] = df['Round'].astype(str)
df['Country'] = df['Country'].astype(str)

countries = ['GHA', 'RWA', 'UGA', 'CIV', 'KEN']
benchmark_rounds = {
    "benchmark_baseline": '0',
    "benchmark_phone_survey_1": '1',
    "benchmark_phone_survey_2": '2',
    "benchmark_phone_survey_3": '3',
    "benchmark_endline": '100',
    "benchmark_post-program_survey_2": '102'
}
round_order = list(benchmark_rounds.values())

for country in countries:
    df_country = df[df['Country'] == country].copy()

    for bench in benchmark_rounds:
        df_country.loc[bench] = np.nan

    benchmark_map = {
        bench: df_country[df_country['Round'] == rnd]
        for bench, rnd in benchmark_rounds.items()
    }

    for col in numerical:
        df_country[col] = pd.to_numeric(df_country[col], errors='coerce')
    for bench, survey_df in benchmark_map.items():
        for col in numerical:
            df_country.at[bench, col] = survey_df[col].mean()
    display(df_country.loc[benchmark_rounds.keys(), numerical])

    for col in ordered_categorical:
        df_country[col] = pd.to_numeric(df_country[col], errors='coerce')
    for bench, survey_df in benchmark_map.items():
        for col in ordered_categorical:
            df_country.at[bench, col] = survey_df[col].mean()
    display(df_country.loc[benchmark_rounds.keys(), ordered_categorical])

    for col in binary:
        if col in df_country.columns:
            df_country[col] = pd.to_numeric(df_country[col], errors='coerce')

    for bench, survey_df in benchmark_map.items():
        for col in binary_neg:
            if col in survey_df.columns:
                total = survey_df[col].isin([1, 2]).sum()
                count = (survey_df[col] == 2).sum()
                df_country.at[bench, col] = count / total if total > 0 else np.nan

        for col in binary_pos:
            if col in survey_df.columns:
                total = survey_df[col].isin([1, 2]).sum()
                count = (survey_df[col] == 1).sum()
                df_country.at[bench, col] = count / total if total > 0 else np.nan

    display(df_country.loc[benchmark_rounds.keys(), binary_neg + binary_pos])

    for var in categorical:
        if var not in df.columns:
            continue

        data_list = []

        for rnd in round_order:
            df_round = df_country[df_country['Round'] == rnd]
            if df_round[var].notna().sum() == 0:
                continue

            val_counts = df_round[var].value_counts(normalize=True).mul(100).round(1)
            for val, pct in val_counts.items():
                data_list.append({'Round': rnd, 'Value': val, 'Percentage': pct})

        if not data_list:
            continue

        plot_df = pd.DataFrame(data_list)
        heatmap_data = plot_df.pivot(index='Value', columns='Round', values='Percentage').fillna(0)

        for r in round_order:
            if r not in heatmap_data.columns:
                heatmap_data[r] = 0
        heatmap_data = heatmap_data[round_order]

        plt.figure(figsize=(10, max(3, 0.4 * len(heatmap_data))))
        sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap='YlGnBu', cbar_kws={'label': 'Percentage (%)'})
        plt.title(f"[{country}] Distribution of '{var}' per Round")
        plt.xlabel("Round")
        plt.ylabel("Value")
        plt.tight_layout()
        plt.show()

    for col in multiple_choice:
        round_counts = {}

        for rnd in round_order:
            df_round = df_country[df_country['Round'] == rnd]
            selections = Counter()

            for entry in df_round[col].dropna():
                parts = str(entry).split("\\")
                for part in parts:
                    part = part.strip()
                    if part:
                        try:
                            processed = int(float(part)) if part.replace('.', '', 1).isdigit() else part
                            selections[str(processed)] += 1
                        except:
                            selections[str(part)] += 1

            total = sum(selections.values())
            if total == 0:
                continue

            pct_series = pd.Series({k: v / total * 100 for k, v in selections.items()})
            round_counts[rnd] = pct_series.round(1)

        if not round_counts:
            continue

        heatmap_df = pd.DataFrame(round_counts).fillna(0)

        for r in round_order:
            if r not in heatmap_df.columns:
                heatmap_df[r] = 0
        heatmap_df = heatmap_df[round_order]

        plt.figure(figsize=(10, max(3, 0.4 * len(heatmap_df))))
        sns.heatmap(heatmap_df, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label': 'Percentage (%)'})
        plt.title(f"[{country}] Distribution of '{col}' values per Round")
        plt.xlabel("Round")
        plt.ylabel("Value")
        plt.tight_layout()
        plt.show()


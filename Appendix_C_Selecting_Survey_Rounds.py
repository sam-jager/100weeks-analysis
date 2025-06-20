survey_counts_per_round = {}

rounds_for_categorical_index = sorted(df['Round'].dropna().unique(), key=lambda x: int(x) if str(x).isdigit() else float('inf'))

for r in rounds_for_categorical_index:
    df_round = df[df['Round'] == r].copy()
    total_rows_in_round = df_round.shape[0]
    survey_counts_per_round[r] = total_rows_in_round

survey_counts_series = pd.Series(survey_counts_per_round)

survey_counts_series.index = pd.CategoricalIndex(survey_counts_series.index, categories=rounds_for_categorical_index, ordered=True)

survey_counts_series = survey_counts_series.sort_index()

plt.figure(figsize=(10, 6))
survey_counts_series.plot(kind='bar', color='forestgreen')
plt.title('Total Number of Surveys Conducted per Round')
plt.xlabel('Round')
plt.ylabel('Amount of Surveys Conducted')
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

cols = [col for col in binary + categorical + ordered_categorical + numerical + multiple_choice if col in df.columns]
rounds = sorted(df['Round'].dropna().unique(), key=lambda x: int(x) if str(x).isdigit() else float('inf'))

summary = {}
for r in map(str, rounds):
    df_r = df[df['Round'] == r]
    if df_r.empty:
        summary[r] = 0
    else:
        total_cells = df_r.shape[0] * len(cols)
        missing = df_r[cols].isnull().sum().sum()
        summary[r] = (missing / total_cells) * 100 if total_cells else 0

pd.Series(summary).plot(kind='bar', figsize=(10, 6), color='skyblue')
plt.title('Percentage Missing Values per Round')
plt.xlabel('Round')
plt.ylabel('Percentage Missing (%)')
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.tight_layout()
plt.show()



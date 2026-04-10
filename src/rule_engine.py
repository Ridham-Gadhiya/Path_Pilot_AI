def filter_by_level(df, user_level):
    return df[
        (df["level_num"] >= user_level) &
        (df["level_num"] <= user_level + 1)
    ]


def filter_by_duration(df, duration):
    return df[df["duration_category"] == duration]


def filter_by_language(df, language):
    return df[df["language"] == language.lower()]
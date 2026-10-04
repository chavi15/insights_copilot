FEW_SHOT_EXAMPLES = [
    {
        "question": "What is the average call duration in minutes for each channel?",
        "sql": "SELECT channel, ROUND(AVG(duration_minutes), 2) AS avg_duration_minutes\nFROM fact_calls\nGROUP BY channel\nORDER BY channel",
    },
    {
        "question": "How many units were sold for each therapeutic area in 2024?",
        "sql": (
            "SELECT p.therapeutic_area, SUM(s.units) AS total_units\n"
            "FROM fact_sales s\n"
            "JOIN dim_product p ON s.product_id = p.product_id\n"
            "JOIN dim_date d ON s.date_id = d.date_id\n"
            "WHERE d.year = 2024\n"
            "GROUP BY p.therapeutic_area\n"
            "ORDER BY p.therapeutic_area"
        ),
    },
    {
        "question": "Show the top 2 specialties by number of calls within each channel.",
        "sql": (
            "WITH calls_by AS (\n"
            "  SELECT c.channel, h.specialty, COUNT(*) AS call_count\n"
            "  FROM fact_calls c\n"
            "  JOIN dim_hcp h ON c.hcp_id = h.hcp_id\n"
            "  GROUP BY c.channel, h.specialty\n"
            "), ranked AS (\n"
            "  SELECT channel, specialty, call_count,\n"
            "         ROW_NUMBER() OVER (PARTITION BY channel ORDER BY call_count DESC, specialty) AS rn\n"
            "  FROM calls_by\n"
            ")\n"
            "SELECT channel, specialty, call_count\n"
            "FROM ranked\n"
            "WHERE rn <= 2\n"
            "ORDER BY channel, rn"
        ),
    },
]

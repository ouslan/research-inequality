import duckdb


def get_conn(db_path: str) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(db_path)


def init_dp03_table(db_path: str) -> None:
    conn = get_conn(db_path=db_path)

    conn.sql(
        """
        CREATE TABLE IF NOT EXISTS "DP03Table" (
            year INTEGER,
            zipcode VARCHAR(5),
            total_population INTEGER,
            total_house INTEGER,
            inc_less_10k INTEGER,
            inc_10k_15k INTEGER,
            inc_15k_25k INTEGER,
            inc_25k_35k INTEGER,
            inc_35k_50k INTEGER,
            inc_50k_75k INTEGER,
            inc_75k_100k INTEGER,
            inc_100k_150k INTEGER,
            inc_150k_200k INTEGER,
            inc_more_200k INTEGER
            );
        """
    )


def init_wb_table(db_path: str) -> None:
    conn = get_conn(db_path=db_path)

    conn.sql(
        """
        CREATE TABLE IF NOT EXISTS "WbTable" (
            year INTEGER,
            country STRING,
            fertility_rate FLOAT,
            gdp_per_capita FLOAT,
            government_expenditure FLOAT,
            trade FLOAT,
            life_expectancy FLOAT,
            gdp_per_capita_growth FLOAT,
            inflation FLOAT,
            population_growth FLOAT,
            school_primary FLOAT,
            school_secondary FLOAT,
            school_tertiary FLOAT,
            gross_capital_formation FLOAT,
            rule_of_law FLOAT,
            control_of_corruption FLOAT,
            regulatory_quality FLOAT,
            political_stability FLOAT,
            voice FLOAT,
            government_effect FLOAT
            );
        """
    )

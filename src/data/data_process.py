import polars.selectors as cs
from src.data.data_pull import DataPull
import polars as pl


class DataClean(DataPull):
    def __init__(
        self,
        saving_dir: str = "data/",
        database_file: str = "data.ddb",
        log_file: str = "data_process.log",
    ):
        super().__init__(saving_dir, database_file, log_file)

    def raw_dataset(self):
        df = self.pull_wb()
        df = df.filter(pl.col("year") >= 1990)
        data = pl.read_csv(f"{self.saving_dir}external/countries.csv")
        df = df.join(data, on="country", how="inner", validate="m:1")
        df = df.select(
            pl.col(
                "year",
                "country",
                "fertility_rate",
                "gdp_per_capita",
                "government_expenditure",
                "trade",
                "life_expectancy",
                "gdp_per_capita_growth",
                "inflation",
                "population_growth",
                "gross_capital_formation",
                "domestic_bank",
                "advanced_economy",
                "rule_of_law",
                "control_of_corruption",
                "regulatory_quality",
                "political_stability",
                "voice",
                "government_effect",
            )
        )
        return df

    def get_remove_countries(self):
        def _gen_dummy(columns: list):
            for cols in columns:
                yield (
                    pl.when(pl.col(cols).is_null())
                    .then(1)
                    .otherwise(0)
                    .name.suffix("_dummy")
                )

        df = self.raw_dataset()
        columns = [
            "fertility_rate",
            "gdp_per_capita",
            "government_expenditure",
            "trade",
            "life_expectancy",
            "gdp_per_capita_growth",
            "inflation",
            "population_growth",
            "gross_capital_formation",
            "domestic_bank",
        ]

        df = df.with_columns(_gen_dummy(columns=columns))
        country_df = df.group_by(["country"]).agg(cs.ends_with("_dummy").mean())

        country_df = country_df.with_columns(
            total_pct=pl.sum_horizontal(cs.ends_with("_dummy")) * 100 / 12
        )
        min_countries = (
            country_df.filter(pl.col("total_pct") <= 5)
            .select(pl.col("country"))
            .to_series()
            .to_list()
        )
        return min_countries

    def dataset(self):
        df = self.raw_dataset()
        min_countries = self.get_remove_countries()
        extra_contries = [
            "Iraq",
            "Malta",
            "Slovak Republic",
            "Slovenia",
            "Guinea",
            "Mozambique",
            "Namibia",
            "Sierra Leone",
            "Solomon Islands",
            "Albania",
            "Armenia",
            "Austria",
            "Azerbaijan",
            "Belarus",
            "Belgium",
            "Brunei Darussalam",
            "Bulgaria",
            "Cambodia",
            "Cyprus",
            "Finland",
            "France",
            "Georgia",
            "Germany",
            "Greece",
            "Ireland",
            "Italy",
            "Kazakhstan",
            "Kuwait",
            "Kyrgyz Republic",
            "Luxembourg",
            "Moldova",
            "Mongolia",
            "North Macedonia",
            "Portugal",
            "Spain",
            "Ukraine",
            "Benin",
            "Nicaragua",
            "Oman",
            "Romania",
            "Uganda",
            "Aruba",
            "Haiti",
            "Hungary",
            "Poland",
            "Qatar",
        ]

        df = df.filter(pl.col("country").is_in(min_countries))
        df = df.filter(~pl.col("country").is_in(extra_contries))

        data = df.to_pandas()
        data = data.sort_values(["year", "country"])
        columns = [
            "fertility_rate",
            "gdp_per_capita",
            "government_expenditure",
            "trade",
            "life_expectancy",
            "gdp_per_capita_growth",
            "inflation",
            "population_growth",
            "gross_capital_formation",
            "domestic_bank",
        ]
        for col in columns:
            data[col] = data.groupby("country")[col].transform(
                lambda group: group.interpolate(method="linear")
            )
        data = data.sort_values(["country", "year"])
        data = data[data["year"] >= 1990].reset_index(drop=True)
        data = pl.from_pandas(data)
        data = data.filter(pl.col("advanced_economy") == 0)
        data = df.with_columns(
            iq=(
                pl.col("rule_of_law")
                + pl.col("control_of_corruption")
                + pl.col("regulatory_quality")
                + pl.col("political_stability")
                + pl.col("voice")
                + pl.col("government_effect")
            )
            / 6
        )
        data = data.filter(pl.col("advanced_economy") == 0)
        IQ = data.select(pl.col("iq").median()).item()

        data = data.group_by("country").agg(iq=pl.col("iq").mean())
        data = data.with_columns(
            iq_dummy=pl.when(pl.col("iq") >= IQ).then(1).otherwise(0)
        )
        dataset = df.join(data, on="country", how="inner", validate="m:1")
        dataset = df.with_columns(
            year_id=pl.col("year").rank(method="min").over("country"),
            county_id=pl.col("country").rank(method="min").over("year"),
        )

        return dataset

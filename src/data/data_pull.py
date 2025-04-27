import logging
import os
from datetime import datetime
from requests.exceptions import HTTPError
import polars as pl
import pandas as pd
import requests
import world_bank_data as wb
from tqdm import tqdm

from ..models import get_conn, init_dp03_table, init_wb_table


class DataClean:
    def __init__(
        self,
        saving_dir: str = "data/",
        database_file: str = "data.ddb",
        log_file: str = "data_process.log",
    ):
        self.saving_dir = saving_dir
        self.data_file = database_file
        self.conn = get_conn(self.data_file)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%d-%b-%y %H:%M:%S",
            filename=log_file,
        )
        # Check if the saving directory exists
        if not os.path.exists(self.saving_dir + "raw"):
            os.makedirs(self.saving_dir + "raw")
        if not os.path.exists(self.saving_dir + "processed"):
            os.makedirs(self.saving_dir + "processed")
        if not os.path.exists(self.saving_dir + "external"):
            os.makedirs(self.saving_dir + "external")

    def pull_query(self, params: list, year: int) -> pl.DataFrame:
        # prepare custom census query
        param = ",".join(params)
        base = "https://api.census.gov/data/"
        flow = "/acs/acs5/profile"
        url = f"{base}{year}{flow}?get={param}&for=zip%20code%20tabulation%20area:*&in=state:72"
        df = pl.DataFrame(requests.get(url).json())

        # get names from DataFrame
        names = df.select(pl.col("column_0")).transpose()
        names = names.to_dicts().pop()
        names = dict((k, v.lower()) for k, v in names.items())

        # Pivot table
        df = df.drop("column_0").transpose()
        return df.rename(names).with_columns(year=pl.lit(year))

    def pull_dp03(self) -> pl.DataFrame:
        if "DP03Table" not in self.conn.sql("SHOW TABLES;").df().get("name").tolist():
            init_dp03_table(self.data_file)
        for _year in range(2012, datetime.now().year):
            if (
                self.conn.sql(f"SELECT * FROM 'DP03Table' WHERE year={_year}")
                .df()
                .empty
            ):
                try:
                    logging.info(f"pulling {_year} data")
                    tmp = self.pull_query(
                        params=[
                            "DP03_0001E",
                            "DP03_0051E",
                            "DP03_0052E",
                            "DP03_0053E",
                            "DP03_0054E",
                            "DP03_0055E",
                            "DP03_0056E",
                            "DP03_0057E",
                            "DP03_0058E",
                            "DP03_0059E",
                            "DP03_0060E",
                            "DP03_0061E",
                        ],
                        year=_year,
                    )
                    tmp = tmp.rename(
                        {
                            "dp03_0001e": "total_population",
                            "dp03_0051e": "total_house",
                            "dp03_0052e": "inc_less_10k",
                            "dp03_0053e": "inc_10k_15k",
                            "dp03_0054e": "inc_15k_25k",
                            "dp03_0055e": "inc_25k_35k",
                            "dp03_0056e": "inc_35k_50k",
                            "dp03_0057e": "inc_50k_75k",
                            "dp03_0058e": "inc_75k_100k",
                            "dp03_0059e": "inc_100k_150k",
                            "dp03_0060e": "inc_150k_200k",
                            "dp03_0061e": "inc_more_200k",
                        }
                    )
                    tmp = tmp.rename({"zip code tabulation area": "zipcode"}).drop(
                        ["state"]
                    )
                    tmp = tmp.with_columns(pl.all().exclude("zipcode").cast(pl.Int64))
                    self.conn.sql("INSERT INTO 'DP03Table' BY NAME SELECT * FROM tmp")
                    logging.info(f"succesfully inserting {_year}")
                except:
                    logging.warning(f"The ACS for {_year} is not availabe")
                    continue
            else:
                logging.info(f"data for {_year} is in the database")
                continue
        return self.conn.sql("SELECT * FROM 'DP03Table';").pl()

    def pull_wb(self) -> pl.DataFrame:
        # Get the list of years in db
        if "WbTable" not in self.conn.sql("SHOW TABLES;").df().get("name").tolist():
            init_wb_table(self.data_file)

        for _year in range(1960, datetime.now().year):
            if (
                self.conn.sql(f"SELECT * FROM 'WbTable' WHERE year={_year}")
                .df()
                .empty
            ):
                try:
                    params = [
                        "SP.DYN.TFRT.IN",
                        "NY.GDP.PCAP.KD",
                        "NE.CON.GOVT.ZS",
                        "NE.TRD.GNFS.ZS",
                        "SP.DYN.LE00.IN",
                        "NY.GDP.PCAP.KD.ZG",
                        "FP.CPI.TOTL.ZG",
                        "SP.POP.GROW",
                        "SE.PRM.ENRR",
                        "SE.SEC.ENRR",
                        "SE.TER.ENRR",
                        "NE.GDI.TOTL.ZS",
                        "RL.PER.RNK",
                        "CC.PER.RNK",
                        "RQ.PER.RNK",
                        "PV.PER.RNK",
                        "VA.PER.RNK",
                        "GE.PER.RNK"


                    ]
                    rename = {
                        "SP.DYN.TFRT.IN": "fertility_rate",
                        "NY.GDP.PCAP.KD": "gdp_per_capita",
                        "NE.CON.GOVT.ZS": "government_expenditure",
                        "NE.TRD.GNFS.ZS": "trade",
                        "SP.DYN.LE00.IN": "life_expectancy",
                        "NY.GDP.PCAP.KD.ZG": "gdp_per_capita_growth",
                        "FP.CPI.TOTL.ZG": "inflation",
                        "SP.POP.GROW": "population_growth",
                        "SE.PRM.ENRR": "school_primary",
                        "SE.SEC.ENRR": "school_secondary",
                        "SE.TER.ENRR": "school_tertiary",
                        "NE.GDI.TOTL.ZS": "gross_capital_formation",
                        "RL.PER.RNK": "rule_of_law",
                        "CC.PER.RNK": "control_of_corruption",
                        "RQ.PER.RNK": "regulatory_quality",
                        "PV.PER.RNK": "political_stability",
                        "VA.PER.RNK": "voice",
                        "GE.PER.RNK": "government_effect"




                        
                    }
                    df = self.wb_data(params=params, year=_year)
                    df = df.rename(rename)
                    self.conn.sql("INSERT INTO 'WbTable' BY NAME SELECT * FROM df")
                    # Logging
                    logging.info(
                        f"Successfully inserted world bank data for year {_year}"
                    )
                except HTTPError:
                    logging.warning(
                        f"Could not insert world bank data for year {_year}"
                    )

            else:
                logging.info(f"Data for year {_year} already exists in gnitable")
                continue
        return self.conn.sql(f"SELECT * FROM 'WbTable'").pl()

    def wb_data(self, params: list, year: int) -> pl.DataFrame:
        df = pl.DataFrame(wb.get_countries())
        df = df.select(pl.col("name")).rename({"name": "country"})
        for param in params:
            tmp = wb.get_series(
                param,
                simplify_index=True,
                date=str(year),
            )
            tmp = pl.DataFrame(pd.DataFrame(tmp).reset_index()).rename(
                {"Country": "country"}
            )
            df = df.join(tmp, on="country", how="inner", validate="1:1")
            df = df.with_columns(year=pl.lit(year))
        return df

    def pull_file(self, url: str, filename: str, verify: bool = True) -> None:
        """
        Pulls a file from a URL and saves it in the filename. Used by the class to pull external files.

        Parameters
        ----------
        url: str
            The URL to pull the file from.
        filename: str
            The filename to save the file to.
        verify: bool
            If True, verifies the SSL certificate. If False, does not verify the SSL certificate.

        Returns
        -------
        None
        """
        chunk_size = 10 * 1024 * 1024

        with requests.get(url, stream=True, verify=verify) as response:
            total_size = int(response.headers.get("content-length", 0))

            with tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc="Downloading",
            ) as bar:
                with open(filename, "wb") as file:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            file.write(chunk)
                            bar.update(
                                len(chunk)
                            )  # Update the progress bar with the size of the chunks

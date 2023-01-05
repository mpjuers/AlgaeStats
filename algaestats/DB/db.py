#!/usr/bin/env python3
# Copyright 2023 Neko Juers

import sqlalchemy as sqa
import pandas as pd


class FlowCamConnection:
    def __init__(
        self,
        local_db="sqlite:///localdb.db",
        flowcam_db="mariadb+mariadbconnector://readonly:cobreadonly@10.111.100.9/flowdb?charset=utf8mb4",
    ):
        """
        Instantiates connection to readonly flowcam database.

        local_db (str): The URL of the local database to connect to.
        flowcam_db (str): The URL of the flowcam read-only db to connect to.
        """
        self.engines = {
            "flowdb": sqa.create_engine(flowcam_db),
            "localdb": sqa.create_engine(local_db),
        }
        self.metadata = {
            "flowdb": sqa.MetaData(),
            "localdb": sqa.MetaData(),
        }
        return None

    def _create_table(self):
        tab = sqa.Table(
            "table",
            self.metadata["localdb"],
            sqa.Column("id", sqa.Integer, primary_key=True),
            sqa.Column("default_classification", sqa.Integer),
            sqa.Column("operator_classification", sqa.Integer),
        )
        self.metadata["localdb"].create_all(self.engines["localdb"])
        return None

    def _query(self, table_name="particle_property", query=None):
        self._connect()
        table = sqa.Table(
            table_name,
            self.metadata["flowdb"],
            autoload=True,
            autoload_with=self.engines["flowdb"],
        )
        if not query:
            # get first ten particles
            query = sqa.select([table]).where(
                table.columns.particle <= 10
            )
        out = pd.DataFrame(self.connections["flowdb"].execute(query).fetchall())
        self._disconnect()
        return out

    def _connect(self):
        self.connections = {
            "flowdb": self.engines["flowdb"].connect(),
            "localdb": self.engines["localdb"].connect(),
        }
        return None

    def _disconnect(self):
        for engine in self.engines.values():
            engine.dispose()
        return None


if __name__ == "__main__":
    c = FlowCamConnection()

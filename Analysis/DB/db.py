#!/usr/bin/env python3
# Copyright 2023 Neko Juers

import sqlalchemy as sqa
import pandas as pd

class FlowCamConnection:
    def __init__(
        self,
        local_db,
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
        return self

    def _query(query=None):
        self._connect()
        if not query:
            query = sqa.select(
                [
                    sqa.Table(
                        "particle_property",
                        self.metadata["flowdb"],
                        autoload=True,
                        autoload_with=self.engines["flowdb"],
                    )
                ]
            )
        out = pd.DataFrame(self.connections["flowdb"].execute(query).fetchall())
        self._disconnect()
        return out

    def _connect():
        self.connections = {
            "flowdb": self.engines["flowdb"].connect()
            "localdb": self.engines["localdb"].connect()
        }
        return None

    def _disconnect():
        for engine in self.engines.values():
            engine.dispose()
        return None

if __name__ == "__main__":
    c = FlowCamConnection()
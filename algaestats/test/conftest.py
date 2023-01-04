#!/usr/bin/env python3
# Copyright 2023 Neko Juers

from pickle import load

import pytest as pt

import algaestats as as

@pt.fixture
def fixture_query_null():
    """
    Default query parameters.
    """
    with open("./TestAssest/test_query_1.pickle", "rb") as file:
        return load(file)

@pt.fixture
def fixture_connection():
    """
    
    """

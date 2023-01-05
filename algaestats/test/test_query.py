#!/usr/bin/env python3
# Copyright 2023 Neko Juers

import pytest as pt

def test_query_null(fixture_query_null, fixture_connection):
    assert fixture_connection._query().equals(fixture_query_null)

#!/bin/bash
coverage run -m pytest
coverage report
coverage-badge -o coverage.svg

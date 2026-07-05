"""
YAML → BoTorch parameter-space loading utilities.

Parses the optimization config (e.g. ``config/minimal.yml``), converting each
parameter's mathematical interval string into safe inclusive/exclusive bounds
and assembling them into a :class:`~opt.params.ParameterSpace`.
"""

import re
import yaml
from pathlib import Path
from opt.params import Parameter, ParameterSpace

# Machine epsilon for BoTorch exclusive bounds
EPSILON = 1e-6

def parse_botorch_bounds_from_string(interval_str):
    """
    Parses a mathematical interval string (e.g., "(0.0, 1.0]") and 
    returns inclusive bounds safely adjusted for BoTorch.
    """
    # Regex to capture: (1) Left bracket, (2) Lower number, (3) Upper number, (4) Right bracket
    # It safely handles spaces and scientific notation (like 1e-4)
    pattern = r"^([\[\(])\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*,\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*([\]\)])$"
    match = re.match(pattern, interval_str.strip())
    
    if not match:
        raise ValueError(f"Invalid interval format: '{interval_str}'. Expected format like '[0.0, 1.0)'")
    
    left_bracket, lower_str, upper_str, right_bracket = match.groups()
    
    lower_val = float(lower_str)
    upper_val = float(upper_str)
    
    # Apply Epsilon for Exclusive Bounds '(' or ')'
    final_lower = lower_val + EPSILON if left_bracket == '(' else lower_val
    final_upper = upper_val - EPSILON if right_bracket == ')' else upper_val
    
    return final_lower, final_upper

def read_config_file(file_path: str | Path):
    """Load and parse a YAML config file into a Python dict.

    Parameters
    ----------
    file_path : str | pathlib.Path
        Path to the YAML config file.

    Returns
    -------
    dict
        The parsed config contents.
    """
    content = None
    with open(file_path, "r") as f:
        content = yaml.safe_load(f)
    
    return content

def build_parameter_space(config_dict):
    """Turn a parsed config dict into a :class:`ParameterSpace`.

    Each entry under the config's ``parameters`` key must define a ``default``
    value and a ``range`` interval string (e.g. ``"(0.0, 6]"``). The interval
    is parsed into BoTorch-safe bounds.

    Parameters
    ----------
    config_dict : dict
        Parsed config, expected to contain a ``"parameters"`` mapping.

    Returns
    -------
    opt.params.ParameterSpace
        The assembled, optimizable parameter space.

    Raises
    ------
    ValueError
        If any parameter is missing the required ``range`` key.
    """
    parameters = []
    for name, details in config_dict["parameters"].items():
        if "range" not in details:
            raise ValueError(f"Parameter '{name}' is missing the required 'range' key.")
        
        lower, upper = parse_botorch_bounds_from_string(details["range"])
        param = Parameter(name=name, default=details["default"], lower=lower, higher=upper)
        parameters.append(param)
    
    return ParameterSpace(parameters)
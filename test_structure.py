#!/usr/bin/env python3
"""Test script to verify the module structure."""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from difffeaspump.core import run_differentiable_feasibility_pump
    print("✓ Successfully imported run_differentiable_feasibility_pump")
    
    from difffeaspump.core.utils import get_integer_vars, get_var_bounds
    print("✓ Successfully imported utils functions")
    
    from difffeaspump.core.losses import integrality_loss_integer
    print("✓ Successfully imported losses functions")
    
    from difffeaspump.core.rounding import round_integer_vars
    print("✓ Successfully imported rounding functions")
    
    print("\n✓ All modules imported successfully!")
    print("The code structure is working correctly.")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Please install dependencies with: pip install -e .")
    
except Exception as e:
    print(f"✗ Error: {e}") 
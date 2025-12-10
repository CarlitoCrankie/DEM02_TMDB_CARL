"""
pipeline.py
===========

Executes in order:
1. Extract - Fetch data from TMDb API
2. Transform/Clean - Process and clean raw data
3. Load/Analyze - Calculate KPIs and perform analysis
4. Visualize - Generate all charts and graphs

Usage:
    python pipeline.py
"""
# This file is AI generated and reviewd as other normal imports and runpy where giving too many issues.
import sys
import os
import importlib.util
from datetime import datetime


# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Script paths (in order of execution)
SCRIPTS = {
    'extraction': 'data_extraction.py',
    'cleaning': 'data_cleaning.py',
    'analysis': 'analysis.py',
    'visualization': 'visualization.py'
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_and_run_script(script_name, script_path):
    
    try:
        # Create a module specification
        spec = importlib.util.spec_from_file_location(script_name, script_path)
        
        if spec is None:
            print(f"✗ Error: Could not load {script_path}")
            return False
        
        # Create the module from the specification
        module = importlib.util.module_from_spec(spec)
        
        # Add to sys.modules so imports work correctly
        sys.modules[script_name] = module
        
        # Execute the module (runs all top-level code)
        spec.loader.exec_module(module)
        
        # Now call the main() function if it exists
        if hasattr(module, 'main'):
            module.main()
            return True
        else:
            print(f"⚠️  Warning: {script_path} has no main() function")
            return False
            
    except Exception as e:
        print(f"✗ Error executing {script_path}:")
        print(f"  {type(e).__name__}: {e}")
        return False


def check_script_exists(script_path):
    """
    Check if a script file exists.
    
    Args:
        script_path (str): Path to check
        
    Returns:
        bool: True if exists, False otherwise
    """
    if not os.path.exists(script_path):
        print(f"✗ Error: Script not found: {script_path}")
        return False
    return True


def print_separator(title):
    """Print a nice separator with title."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_pipeline():
    """
    Execute the complete ETL pipeline.
    
    ETL Flow:
    - Extract: Fetch data from TMDb API → data/raw/
    - Transform: Clean and process data → data/processed/
    - Load: Analyze and calculate KPIs → data/analysis/
    - Visualize: Generate charts → visualizations/
    """
    
    # Pipeline start
    print_separator("TMDb MOVIE ANALYSIS PIPELINE - START")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Working directory: {BASE_DIR}\n")
    
    # Track success/failure
    results = {}
    
    # ========================================================================
    # STEP 1: EXTRACT - Data Extraction from TMDb API
    # ========================================================================
    print_separator("STEP 1/4: DATA EXTRACTION (Extract)")
    
    extraction_script = os.path.join(BASE_DIR, SCRIPTS['extraction'])
    
    if check_script_exists(extraction_script):
        print(f"Running: {SCRIPTS['extraction']}")
        results['extraction'] = load_and_run_script('extraction_module', extraction_script)
        
        if results['extraction']:
            print("\n✅ Data extraction completed successfully!")
        else:
            print("\n❌ Data extraction failed!")
            print("   Cannot proceed to next steps.")
            return False
    else:
        return False
    
    # ========================================================================
    # STEP 2: TRANSFORM/CLEAN - Data Cleaning and Transformation
    # ========================================================================
    print_separator("STEP 2/4: DATA CLEANING (Transform)")
    
    cleaning_script = os.path.join(BASE_DIR, SCRIPTS['cleaning'])
    
    if check_script_exists(cleaning_script):
        print(f"Running: {SCRIPTS['cleaning']}")
        results['cleaning'] = load_and_run_script('cleaning_module', cleaning_script)
        
        if results['cleaning']:
            print("\n✅ Data cleaning completed successfully!")
        else:
            print("\n❌ Data cleaning failed!")
            print("   Cannot proceed to analysis.")
            return False
    else:
        return False
    
    # ========================================================================
    # STEP 3: LOAD/ANALYZE - KPI Calculation and Analysis
    # ========================================================================
    print_separator("STEP 3/4: DATA ANALYSIS (Load)")
    
    analysis_script = os.path.join(BASE_DIR, SCRIPTS['analysis'])
    
    if check_script_exists(analysis_script):
        print(f"Running: {SCRIPTS['analysis']}")
        results['analysis'] = load_and_run_script('analysis_module', analysis_script)
        
        if results['analysis']:
            print("\n✅ Analysis completed successfully!")
        else:
            print("\n❌ Analysis failed!")
            print("   Visualization may not work properly.")
            # Continue anyway - maybe some data exists
    else:
        return False
    
    # ========================================================================
    # STEP 4: VISUALIZE - Generate Charts and Graphs
    # ========================================================================
    print_separator("STEP 4/4: DATA VISUALIZATION")
    
    viz_script = os.path.join(BASE_DIR, SCRIPTS['visualization'])
    
    if check_script_exists(viz_script):
        print(f"Running: {SCRIPTS['visualization']}")
        results['visualization'] = load_and_run_script('visualization_module', viz_script)
        
        if results['visualization']:
            print("\n✅ Visualization completed successfully!")
        else:
            print("\n❌ Visualization failed!")
    else:
        return False
    
    # ========================================================================
    # PIPELINE SUMMARY
    # ========================================================================
    print_separator("PIPELINE EXECUTION SUMMARY")
    
    print("Step Results:")
    print("-" * 70)
    for step, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {step.capitalize():15} : {status}")
    
    print("\n" + "-" * 70)
    
    all_success = all(results.values())
    
    if all_success:
        print("\n🎉 ALL STEPS COMPLETED SUCCESSFULLY! 🎉")
        print("\nGenerated outputs:")
        print("  📁 data/raw/               - Raw API data")
        print("  📁 data/processed/         - Cleaned data")
        print("  📁 data/analysis/          - Analysis results")
        print("  📁 visualizations/         - Charts and graphs")
    else:
        print("\n⚠️  PIPELINE COMPLETED WITH ERRORS")
        print("   Check the output above for details.")
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_separator("PIPELINE END")
    
    return all_success


# ============================================================================
# ALTERNATIVE: RUN INDIVIDUAL STEPS
# ============================================================================

def run_step(step_name):
    """
    Run a single pipeline step.
    
    Args:
        step_name (str): One of 'extraction', 'cleaning', 'analysis', 'visualization'
        
    Returns:
        bool: Success status
    """
    if step_name not in SCRIPTS:
        print(f"✗ Error: Unknown step '{step_name}'")
        print(f"  Available steps: {list(SCRIPTS.keys())}")
        return False
    
    script_path = os.path.join(BASE_DIR, SCRIPTS[step_name])
    
    if not check_script_exists(script_path):
        return False
    
    print_separator(f"RUNNING STEP: {step_name.upper()}")
    success = load_and_run_script(f'{step_name}_module', script_path)
    
    if success:
        print(f"\n✅ {step_name.capitalize()} completed!")
    else:
        print(f"\n❌ {step_name.capitalize()} failed!")
    
    return success


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Main entry point with command line argument support."""
    
    # Check if user wants to run a specific step
    if len(sys.argv) > 1:
        step_name = sys.argv[1].lower()
        
        if step_name in ['--help', '-h']:
            print("Usage:")
            print("  python pipeline.py              # Run full pipeline")
            print("  python pipeline.py extraction   # Run extraction only")
            print("  python pipeline.py cleaning     # Run cleaning only")
            print("  python pipeline.py analysis     # Run analysis only")
            print("  python pipeline.py visualization # Run visualization only")
            return
        
        # Run single step
        success = run_step(step_name)
        sys.exit(0 if success else 1)
    
    else:
        # Run full pipeline
        success = run_pipeline()
        sys.exit(0 if success else 1)


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
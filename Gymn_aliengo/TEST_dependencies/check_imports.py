import importlib

def check_package(package_name):
    try:
        package = importlib.import_module(package_name)
        version = package.__version__ if hasattr(package, '__version__') else 'Version not available'
        print(f"{package_name:<12} --> ACTIVE --> Version: {version}")
    except ImportError:
        print(f"{package_name:<12} - NOT active or installed.")
        print("Activate you conda env and set VSCode python interpreter !! (ctrl+shift+p)")

packages_to_check = ['gym', 'gymnasium', 'skrl', 'pybullet']

for package in packages_to_check:
    check_package(package)

import subprocess
import sys

def install_package(package, import_name=None):
    """Cài đặt package nếu chưa có"""
    if import_name is None:
        import_name = package
    try:
        __import__(import_name)
        print(f"✓ {package} đã được cài đặt")
        return True
    except ImportError:
        print(f"📦 Đang cài đặt {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"✓ Đã cài đặt {package}")
            return True
        except Exception as e:
            print(f"⚠️  Lỗi khi cài đặt {package}: {e}")
            return False

def install():
    packages_to_install = [
        ('neuralforecast', 'neuralforecast'),
        ('optuna', 'optuna'),
        ('scikit-learn', 'sklearn'),
        ('scipy', 'scipy')
    ]

    print("🔧 Kiểm tra và cài đặt các thư viện cần thiết...\n")
    for package, import_name in packages_to_install:
        install_package(package, import_name)

    print("\n✓ Hoàn thành kiểm tra/cài đặt thư viện!")
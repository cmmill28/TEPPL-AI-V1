# quick_inventory.py
from pathlib import Path
from collections import Counter

def quick_inventory(docs_path="./documents"):
    """Quick inventory of all files in documents directory."""
    base_path = Path(docs_path)
    
    if not base_path.exists():
        print(f"❌ Directory not found: {docs_path}")
        return
    
    print(f"🔍 Scanning: {base_path.absolute()}")
    print("="*60)
    
    all_files = list(base_path.rglob('*.*'))
    
    # File composition
    file_types = Counter(f.suffix.lower().lstrip('.') for f in all_files if f.suffix)
    
    # Folder composition  
    folders = Counter(f.parent.name for f in all_files)
    
    total_files = len(all_files)
    
    print(f"📈 TOTAL FILES: {total_files:,}")
    print(f"\n📊 BY FILE TYPE:")
    
    for ext, count in file_types.most_common():
        percentage = (count / total_files) * 100
        print(f"  {ext.upper():<6}: {count:>5,} files ({percentage:>5.1f}%)")
    
    print(f"\n📁 BY FOLDER:")
    for folder, count in folders.most_common():
        percentage = (count / total_files) * 100  
        print(f"  {folder:<15}: {count:>5,} files ({percentage:>5.1f}%)")
    
    print("="*60)

if __name__ == "__main__":
    quick_inventory()

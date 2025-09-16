# file_organizer.py
"""
Professional File Organization System for TEPPL Documents
Automatically organizes mixed file types into proper directory structure
"""

import os
import shutil
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import mimetypes

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TEPPLFileOrganizer:
    """Professional file organization system for TEPPL documents"""
    
    def __init__(self, base_documents_path: str = "./documents"):
        self.base_path = Path(base_documents_path)
        self.stats = {
            'processed': 0,
            'skipped': 0,
            'errors': 0,
            'by_type': {}
        }
        
        # File type mappings
        self.file_categories = {
            'pdfs': ['.pdf'],
            'html': ['.html', '.htm'],
            'excel': ['.xlsx', '.xls', '.xlsm', '.csv'],
            'aspx': ['.aspx'],
            'images': ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.svg'],
            'documents': ['.doc', '.docx', '.txt', '.rtf'],
            'archives': ['.zip', '.rar', '.7z', '.tar', '.gz'],
            'other': []  # fallback category
        }
        
        self._setup_directories()
    
    def _setup_directories(self):
        """Create the organized directory structure"""
        for category in self.file_categories.keys():
            category_path = self.base_path / category
            category_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"📁 Ensured directory: {category_path}")
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Generate MD5 hash for file deduplication"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            logger.warning(f"Could not hash {file_path}: {e}")
            return None
    
    def _determine_category(self, file_path: Path) -> str:
        """Determine which category a file belongs to"""
        suffix = file_path.suffix.lower()
        
        for category, extensions in self.file_categories.items():
            if suffix in extensions:
                return category
        
        # Try MIME type detection for unknown extensions
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type:
            if mime_type.startswith('image/'):
                return 'images'
            elif mime_type.startswith('text/'):
                return 'documents'
            elif mime_type == 'application/pdf':
                return 'pdfs'
        
        return 'other'
    
    def _generate_safe_filename(self, original_path: Path, target_dir: Path) -> Path:
        """Generate a safe filename avoiding conflicts"""
        base_name = original_path.stem
        extension = original_path.suffix
        counter = 1
        
        target_path = target_dir / original_path.name
        
        while target_path.exists():
            new_name = f"{base_name}_{counter:03d}{extension}"
            target_path = target_dir / new_name
            counter += 1
        
        return target_path
    
    def organize_directory(self, source_directory: str, move_files: bool = True) -> Dict:
        """
        Organize all files from source directory into categorized structure
        
        Args:
            source_directory: Path to directory containing mixed files
            move_files: If True, move files; if False, copy files
            
        Returns:
            Dictionary with organization statistics
        """
        source_path = Path(source_directory)
        if not source_path.exists():
            raise FileNotFoundError(f"Source directory not found: {source_directory}")
        
        logger.info(f"🚀 Starting file organization from: {source_path}")
        logger.info(f"📋 Operation mode: {'MOVE' if move_files else 'COPY'}")
        
        # Track processed files to avoid duplicates
        processed_hashes = set()
        organization_log = []
        
        # Process all files recursively
        for file_path in source_path.rglob('*'):
            if file_path.is_file():
                try:
                    self._process_single_file(
                        file_path, processed_hashes, organization_log, move_files
                    )
                except Exception as e:
                    logger.error(f"❌ Error processing {file_path}: {e}")
                    self.stats['errors'] += 1
        
        # Save organization log
        self._save_organization_log(organization_log)
        
        logger.info(f"✅ Organization complete! Processed: {self.stats['processed']}, "
                   f"Skipped: {self.stats['skipped']}, Errors: {self.stats['errors']}")
        
        return self.stats
    
    def _process_single_file(self, file_path: Path, processed_hashes: set, 
                           organization_log: list, move_files: bool):
        """Process a single file"""
        
        # Skip hidden files and system files
        if file_path.name.startswith('.') or file_path.name.startswith('~'):
            self.stats['skipped'] += 1
            return
        
        # Check for duplicates
        file_hash = self._get_file_hash(file_path)
        if file_hash and file_hash in processed_hashes:
            logger.info(f"⏭️  Skipping duplicate: {file_path.name}")
            self.stats['skipped'] += 1
            return
        
        # Determine target category
        category = self._determine_category(file_path)
        target_dir = self.base_path / category
        
        # Generate safe target path
        target_path = self._generate_safe_filename(file_path, target_dir)
        
        # Perform the file operation
        try:
            if move_files:
                shutil.move(str(file_path), str(target_path))
                operation = "MOVED"
            else:
                shutil.copy2(str(file_path), str(target_path))
                operation = "COPIED"
            
            # Update statistics
            self.stats['processed'] += 1
            if category not in self.stats['by_type']:
                self.stats['by_type'][category] = 0
            self.stats['by_type'][category] += 1
            
            # Add to processed hashes
            if file_hash:
                processed_hashes.add(file_hash)
            
            # Log the operation
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'operation': operation,
                'source': str(file_path),
                'target': str(target_path),
                'category': category,
                'size_bytes': target_path.stat().st_size,
                'hash': file_hash
            }
            organization_log.append(log_entry)
            
            logger.info(f"✅ {operation}: {file_path.name} → {category}/")
            
        except Exception as e:
            logger.error(f"❌ Failed to process {file_path}: {e}")
            self.stats['errors'] += 1
    
    def _save_organization_log(self, organization_log: list):
        """Save detailed organization log"""
        log_file = self.base_path / "organization_log.json"
        
        complete_log = {
            'organization_date': datetime.now().isoformat(),
            'statistics': self.stats,
            'files': organization_log
        }
        
        try:
            with open(log_file, 'w') as f:
                json.dump(complete_log, f, indent=2)
            logger.info(f"📋 Organization log saved: {log_file}")
        except Exception as e:
            logger.warning(f"Could not save organization log: {e}")
    
    def scan_directory_preview(self, source_directory: str) -> Dict:
        """Preview what would be organized without actually moving files"""
        source_path = Path(source_directory)
        if not source_path.exists():
            raise FileNotFoundError(f"Source directory not found: {source_directory}")
        
        preview = {
            'total_files': 0,
            'by_category': {},
            'large_files': [],  # Files > 100MB
            'potential_duplicates': []
        }
        
        file_hashes = {}
        
        for file_path in source_path.rglob('*'):
            if file_path.is_file() and not file_path.name.startswith('.'):
                preview['total_files'] += 1
                
                # Categorize
                category = self._determine_category(file_path)
                if category not in preview['by_category']:
                    preview['by_category'][category] = 0
                preview['by_category'][category] += 1
                
                # Check size
                size_mb = file_path.stat().st_size / (1024 * 1024)
                if size_mb > 100:
                    preview['large_files'].append({
                        'file': str(file_path),
                        'size_mb': round(size_mb, 2)
                    })
                
                # Check for potential duplicates
                file_hash = self._get_file_hash(file_path)
                if file_hash:
                    if file_hash in file_hashes:
                        preview['potential_duplicates'].append({
                            'files': [file_hashes[file_hash], str(file_path)],
                            'hash': file_hash
                        })
                    else:
                        file_hashes[file_hash] = str(file_path)
        
        return preview


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Organize TEPPL files into proper structure')
    parser.add_argument('source', help='Source directory containing mixed files')
    parser.add_argument('--target', default='./documents', help='Target documents directory')
    parser.add_argument('--copy', action='store_true', help='Copy files instead of moving')
    parser.add_argument('--preview', action='store_true', help='Preview organization without moving files')
    
    args = parser.parse_args()
    
    # Initialize organizer
    organizer = TEPPLFileOrganizer(args.target)
    
    if args.preview:
        print("🔍 PREVIEW MODE - No files will be moved")
        preview = organizer.scan_directory_preview(args.source)
        
        print(f"\n📊 ORGANIZATION PREVIEW")
        print(f"Total files to process: {preview['total_files']}")
        print(f"\nFiles by category:")
        for category, count in preview['by_category'].items():
            print(f"  📁 {category}: {count} files")
        
        if preview['large_files']:
            print(f"\n⚠️  Large files (>100MB): {len(preview['large_files'])}")
            for file_info in preview['large_files'][:5]:  # Show first 5
                print(f"  📄 {Path(file_info['file']).name}: {file_info['size_mb']}MB")
        
        if preview['potential_duplicates']:
            print(f"\n🔄 Potential duplicates: {len(preview['potential_duplicates'])}")
    
    else:
        # Perform actual organization
        print(f"🚀 ORGANIZING FILES")
        print(f"Source: {args.source}")
        print(f"Target: {args.target}")
        print(f"Mode: {'COPY' if args.copy else 'MOVE'}")
        
        # Confirm before proceeding
        response = input("\nProceed with file organization? (y/N): ")
        if response.lower() != 'y':
            print("❌ Operation cancelled")
            return
        
        stats = organizer.organize_directory(args.source, move_files=not args.copy)
        
        print(f"\n✅ ORGANIZATION COMPLETE")
        print(f"Processed: {stats['processed']} files")
        print(f"Skipped: {stats['skipped']} files")
        print(f"Errors: {stats['errors']} files")
        print(f"\nFiles by type:")
        for file_type, count in stats['by_type'].items():
            print(f"  📁 {file_type}: {count} files")


if __name__ == "__main__":
    main()
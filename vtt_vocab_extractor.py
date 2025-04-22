"""
Japanese Vocabulary Extractor for VTT Files
This script extracts Japanese vocabulary from VTT subtitle files, looks up definitions,
and outputs a structured CSV for language learning.
"""
import re
import csv
import argparse
import os
from collections import Counter
from typing import Dict, List, Set, Tuple, Optional

import pykakasi
import requests
import gzip
import xml.etree.ElementTree as ET
import glob
from sudachipy import Dictionary, SplitMode

# --- Constants ---
# JMdict Dictionary
JMDICT_URL = "http://ftp.edrdg.org/pub/Nihongo/JMdict_e.gz"
JMDICT_GZ_FILENAME = "JMdict_e.gz"
JMDICT_XML_FILENAME = "JMdict_e.xml"

# JLPT Data
JLPT_DATA_DIR = "data/jlpt"

# Regex Patterns
TIMESTAMP_PATTERN = re.compile(r"^\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}")
SPEAKER_TAG_PATTERN = re.compile(r"（.*?）")
ANNOTATION_PATTERN = re.compile(r"[<＜《》♫～｢｣()＞]→")
TAG_PATTERN = re.compile(r"<[^>]*>")

# Parts of speech we're interested in extracting
TARGET_POS = {'名詞', '動詞', '形容詞', '副詞', '感動詞'}


class DictionaryManager:
    """Handles JMdict dictionary download and parsing."""
    
    def __init__(self):
        self.jmdict_data = None
        
    def prepare_dictionary(self) -> bool:
        """Download and extract JMdict if needed, then load it."""
        if not self._download_and_extract_jmdict():
            print("Failed to prepare JMdict dictionary file. Cannot proceed.")
            return False
        
        self.jmdict_data = self._load_jmdict()
        return self.jmdict_data is not None
        
    def _download_and_extract_jmdict(self) -> bool:
        """Downloads JMdict_e.gz if needed, and extracts it to XML."""
        if os.path.exists(JMDICT_XML_FILENAME):
            print(f"Found existing dictionary file: {JMDICT_XML_FILENAME}")
            return True

        if not os.path.exists(JMDICT_GZ_FILENAME):
            print(f"Downloading dictionary file from {JMDICT_URL} ...")
            try:
                response = requests.get(JMDICT_URL, stream=True)
                response.raise_for_status()
                with open(JMDICT_GZ_FILENAME, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print("Download complete.")
            except requests.exceptions.RequestException as e:
                print(f"Error downloading dictionary: {e}")
                return False
        else:
            print(f"Found existing compressed dictionary file: {JMDICT_GZ_FILENAME}")

        print(f"Extracting {JMDICT_GZ_FILENAME} to {JMDICT_XML_FILENAME} ...")
        try:
            with gzip.open(JMDICT_GZ_FILENAME, 'rb') as f_in:
                with open(JMDICT_XML_FILENAME, 'wb') as f_out:
                    f_out.write(f_in.read())
            print("Extraction complete.")
            return True
        except Exception as e:
            print(f"Error extracting dictionary: {e}")
            return False
            
    def _load_jmdict(self) -> Optional[Dict[str, str]]:
        """Parses the JMdict XML file and loads entries into a dictionary."""
        print(f"Loading dictionary from {JMDICT_XML_FILENAME} into memory (this may take a moment)...")
        jmdict_data = {}
        try:
            # Use iterparse for memory efficiency with large XML files
            context = ET.iterparse(JMDICT_XML_FILENAME, events=('end',))
            _, root = next(context)  # Get the root element

            for event, elem in context:
                if event == 'end' and elem.tag == 'entry':
                    kanji_elements = [k.text for k in elem.findall('k_ele/keb')]
                    reading_elements = [r.text for r in elem.findall('r_ele/reb')]
                    
                    definitions = []
                    sense_elements = elem.findall('sense')
                    if sense_elements:
                        # Take first 2 senses
                        for sense in sense_elements[:2]:
                            glosses = sense.findall('gloss')
                            # Take first 2 glosses per sense
                            definitions.extend([g.text for g in glosses[:2] 
                                              if g.text and g.get('{http://www.w3.org/XML/1998/namespace}lang', 'eng') == 'eng'])
                        
                    if definitions:
                        definition_str = "; ".join(definitions)
                        # Map all forms (kanji and readings) to the definition string
                        all_forms = set(kanji_elements + reading_elements)
                        for form in all_forms:
                            if form not in jmdict_data:  # Store only the first definition encountered per form
                                jmdict_data[form] = definition_str
                    
                    # Clear the element to save memory
                    elem.clear()
                    
            print(f"Dictionary loaded. Found definitions for {len(jmdict_data)} unique forms.")
            return jmdict_data
        except ET.ParseError as e:
            print(f"Error parsing JMdict XML: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during dictionary loading: {e}")
            return None
    
    def get_definition(self, lemma: str, word: str) -> str:
        """Looks up English definition in the pre-loaded JMdict data, trying lemma then original word."""
        if not self.jmdict_data:
            return "N/A (Dictionary not loaded)"
        
        # 1. Try looking up the lemma provided by SudachiPy
        definition = self.jmdict_data.get(lemma)
        if definition:
            return definition
        
        # 2. If lemma not found AND lemma is different from the surface word, try the surface word
        if lemma != word:
            definition = self.jmdict_data.get(word)
            if definition:
                return definition
                
        # 3. If neither is found
        return "N/A (Not found in JMdict)"


class JLPTData:
    """Manages JLPT level data loading and lookup."""
    
    def __init__(self):
        self.jlpt_mapping = self._load_jlpt_data()
        
    def _load_jlpt_data(self) -> Dict[str, str]:
        """Loads JLPT level data from CSV files and creates a mapping for lookup."""
        print("Loading JLPT level data...")
        jlpt_mapping = {}  # Maps words to JLPT levels
        
        # Find all JLPT data files
        data_files = glob.glob(os.path.join(JLPT_DATA_DIR, "n*.csv"))
        if not data_files:
            print(f"No JLPT data files found in {JLPT_DATA_DIR}")
            return jlpt_mapping
        
        for file_path in data_files:
            # Extract JLPT level from filename (n1.csv -> N1)
            jlpt_level = os.path.basename(file_path).split('.')[0].upper()
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Add both kanji and kana forms to mapping
                        if row.get('kanji') and row['kanji'].strip():
                            jlpt_mapping[row['kanji']] = jlpt_level
                        if row.get('kana') and row['kana'].strip():
                            jlpt_mapping[row['kana']] = jlpt_level
            except Exception as e:
                print(f"Error loading JLPT data from {file_path}: {e}")
        
        print(f"Loaded {len(jlpt_mapping)} JLPT vocabulary items.")
        return jlpt_mapping
        
    def get_jlpt_level(self, word: str) -> str:
        """Looks up JLPT level for a word."""
        return self.jlpt_mapping.get(word, "N/A")


class TextProcessor:
    """Handles text processing and vocabulary extraction."""
    
    def __init__(self):
        self.kakasi = pykakasi.kakasi()
        # Initialize SudachiPy Tokenizer
        try:
            self.tokenizer = Dictionary().create()
            self.sudachi_mode = SplitMode.B  # Mode B for better word segmentation
        except Exception as e:
            print(f"Error initializing SudachiPy tokenizer: {e}")
            print("Ensure sudachipy and sudachidict_core are installed.")
            self.tokenizer = None
    
    def extract_text_from_vtt(self, vtt_path: str) -> List[str]:
        """Extracts Japanese text lines from a VTT file."""
        lines = []
        current_sentence_parts = []

        with open(vtt_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                is_separator = (not line or line == "WEBVTT" or TIMESTAMP_PATTERN.match(line) or line.isdigit())

                if is_separator:
                    if current_sentence_parts:
                        full_sentence = " ".join(current_sentence_parts)
                        cleaned_sentence = self._clean_text(full_sentence)
                        if cleaned_sentence.strip():
                            lines.append(cleaned_sentence.strip())
                        current_sentence_parts = []
                else:
                    cleaned_part = self._clean_text(line)
                    if cleaned_part.strip():
                        current_sentence_parts.append(cleaned_part.strip())

        if current_sentence_parts:
            full_sentence = " ".join(current_sentence_parts)
            cleaned_sentence = self._clean_text(full_sentence)
            if cleaned_sentence.strip():
                lines.append(cleaned_sentence.strip())

        return [line for line in lines if line]
    
    def _clean_text(self, text: str) -> str:
        """Clean a text line by removing tags, annotations, etc."""
        text = SPEAKER_TAG_PATTERN.sub('', text)
        text = TAG_PATTERN.sub('', text)
        text = ANNOTATION_PATTERN.sub('', text)
        # Explicitly remove full-width brackets
        text = re.sub('＜', '', text)
        text = re.sub('＞', '', text)
        return text
    
    def get_reading(self, text: str) -> str:
        """Generates Hiragana reading for Japanese text using pykakasi."""
        result = self.kakasi.convert(text)
        return "".join([item.get('hira', '') for item in result]) if result else text
    
    def is_valid_tokenizer(self) -> bool:
        """Check if the tokenizer is available and valid."""
        return self.tokenizer is not None
        
        
class VocabularyExtractor:
    """Main class for vocabulary extraction and processing."""
    
    def __init__(self):
        self.dict_manager = DictionaryManager()
        self.jlpt_data = JLPTData()
        self.text_processor = TextProcessor()
        
    def process_vtt_file(self, vtt_path: str, csv_path: str) -> bool:
        """Process a VTT file and extract vocabulary to a CSV file."""
        # Prepare dictionary
        if not self.dict_manager.prepare_dictionary():
            return False
            
        # Check tokenizer
        if not self.text_processor.is_valid_tokenizer():
            print("SudachiPy failed to initialize. Cannot proceed.")
            return False
            
        # Extract text from VTT file
        print(f"Processing VTT file: {vtt_path}")
        text_lines = self.text_processor.extract_text_from_vtt(vtt_path)
        print(f"Extracted {len(text_lines)} dialogue lines.")
        
        if not text_lines:
            print("No text lines extracted. Exiting.")
            return False
            
        # Extract vocabulary items
        vocab_items = self._extract_vocabulary(text_lines)
        print(f"Extracted {len(vocab_items)} raw vocabulary items.")
        
        if not vocab_items:
            print("No vocabulary items extracted. Exiting.")
            return False
            
        # Process and filter vocabulary items
        processed_vocab = self._process_vocabulary_items(vocab_items)
        
        # Write to CSV
        return self._write_csv(processed_vocab, csv_path)
        
    def _extract_vocabulary(self, text_lines: List[str]) -> List[Dict]:
        """Extract vocabulary items from text lines."""
        vocab_items = []
        processed_lemmas_in_sentence = set()
        total_lines = len(text_lines)

        print("Tokenizing sentences and extracting vocabulary using SudachiPy...")
        for idx, sentence in enumerate(text_lines):
            print(f"Processing sentence {idx + 1}/{total_lines}", end='\r')
            if not sentence.strip():
                continue
            processed_lemmas_in_sentence.clear()
            
            try:
                # Tokenize using SudachiPy
                tokens = self.text_processor.tokenizer.tokenize(sentence, self.text_processor.sudachi_mode)

                for token in tokens:
                    surface_form = token.surface()
                    pos = token.part_of_speech()[0]  # Get first element (main POS)
                    dictionary_form = token.dictionary_form()

                    # Skip if not a target part of speech, not Japanese, or already processed
                    if (pos not in TARGET_POS or 
                        not re.search(r'[一-龯ァ-ヶｱ-ﾝﾞﾟ]', dictionary_form) or 
                        dictionary_form in processed_lemmas_in_sentence):
                        continue
                        
                    # Get reading, definition, and JLPT level
                    reading = self.text_processor.get_reading(surface_form)
                    definition = self.dict_manager.get_definition(dictionary_form, surface_form)
                    
                    # Look up JLPT level - try dictionary form first, then surface form
                    jlpt_level = self.jlpt_data.get_jlpt_level(dictionary_form)
                    if jlpt_level == "N/A" and dictionary_form != surface_form:
                        jlpt_level = self.jlpt_data.get_jlpt_level(surface_form)
                    
                    vocab_items.append({
                        "lemma": dictionary_form,
                        "reading": reading,
                        "pos": pos,
                        "sentence": sentence,
                        "definition": definition,
                        "jlpt": jlpt_level
                    })
                    processed_lemmas_in_sentence.add(dictionary_form)

            except Exception as e:
                print(f"\nError processing sentence with SudachiPy: '{sentence}'. Error: {e}")
                continue
                
        print("\nFinished processing sentences.")
        return vocab_items
        
    def _process_vocabulary_items(self, vocab_items: List[Dict]) -> List[Dict]:
        """Process and filter vocabulary items."""
        # Count frequency based on lemma (dictionary form) and POS
        item_counter = Counter((item['lemma'], item['pos']) for item in vocab_items)

        # Create a unique list
        unique_vocab = {}
        for item in vocab_items:
            key = (item['lemma'], item['pos'])
            if key not in unique_vocab:
                unique_vocab[key] = {
                    "lemma": item['lemma'],
                    "reading": item['reading'],
                    "pos": item['pos'],
                    "sentence": item['sentence'],
                    "definition": item['definition'],
                    "jlpt": item['jlpt'],
                    "frequency": item_counter[key]
                }

        # Sort by JLPT level and frequency
        sorted_vocab = sorted(unique_vocab.values(), key=self._jlpt_sort_key)
        print(f"Found {len(sorted_vocab)} unique vocabulary items.")

        # Filter out items where definition was not found
        filtered_vocab = [item for item in sorted_vocab if "N/A (Not found in JMdict)" not in item['definition']]
        print(f"Filtered out {len(sorted_vocab) - len(filtered_vocab)} items without definitions. Writing {len(filtered_vocab)} items.")
        
        return filtered_vocab
        
    def _jlpt_sort_key(self, item: Dict) -> Tuple:
        """Sort key that prioritizes N/A first, then N1 through N5, then by frequency."""
        jlpt = item['jlpt']
        # Assign a numeric value to each JLPT level
        if jlpt == "N/A":
            jlpt_value = 0  # N/A comes first
        else:
            # Extract the numeric part and convert to int (N1 -> 1, N5 -> 5)
            jlpt_value = int(jlpt[1:])
        
        # Return a tuple with jlpt_value first, then negative frequency for descending order
        return (jlpt_value, -item['frequency'])
        
    def _write_csv(self, vocab_items: List[Dict], csv_path: str) -> bool:
        """Write vocabulary items to a CSV file."""
        print(f"Writing vocabulary to CSV: {csv_path}")
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['Word (Dictionary Form)', 'Reading', 'Part of Speech', 
                            'English Definition', 'Original Sentence', 'JLPT Level', 'Frequency']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                for item in vocab_items:
                    writer.writerow({
                        'Word (Dictionary Form)': item['lemma'],
                        'Reading': item['reading'],
                        'Part of Speech': item['pos'],
                        'English Definition': item['definition'],
                        'Original Sentence': item['sentence'],
                        'JLPT Level': item['jlpt'],
                        'Frequency': item['frequency']
                    })
            print("CSV export complete.")
            return True
        except Exception as e:
            print(f"Error writing CSV file: {e}")
            return False


def main(vtt_path: str, csv_path: str) -> None:
    """Main function to process a VTT file and extract vocabulary."""
    extractor = VocabularyExtractor()
    extractor.process_vtt_file(vtt_path, csv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract Japanese vocabulary with definitions from a VTT subtitle file."
    )
    parser.add_argument("vtt_file", help="Path to the input VTT file.")
    parser.add_argument("csv_file", help="Path to the output CSV file.")
    
    args = parser.parse_args()
    
    main(args.vtt_file, args.csv_file)
"""
Japanese Vocabulary Extractor for VTT Files

This script extracts Japanese vocabulary from VTT subtitle files, looks up definitions,
and outputs a structured CSV for language learning.
"""
import re
import csv
import argparse
import os
import gzip
import glob
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional, Set

import pykakasi
import requests
from sudachipy import Dictionary, SplitMode

# Constants
JMDICT_URL = "http://ftp.edrdg.org/pub/Nihongo/JMdict_e.gz"
JMDICT_GZ_FILENAME = "JMdict_e.gz"
JMDICT_XML_FILENAME = "JMdict_e.xml"
JLPT_DATA_DIR = "data/jlpt"

# Regex patterns for VTT parsing
TIMESTAMP_PATTERN = re.compile(r"^\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}")
SPEAKER_TAG_PATTERN = re.compile(r"（.*?）")
ANNOTATION_PATTERN = re.compile(r"[<＜《》♫～｢｣()＞]→")
TAG_PATTERN = re.compile(r"<[^>]*>")

# Parts of speech to extract
TARGET_POS = {'名詞', '動詞', '形容詞', '副詞', '感動詞'}

# Parts of speech to exclude (even if they are a subtype of target)
EXCLUDE_POS_SUBTYPES = {
    '助動詞',       # Auxiliary verbs
    '形状詞',       # Adjectival nouns
    '助詞',         # Particles
    '接尾辞',       # Suffixes
    '接頭辞',       # Prefixes
    '動詞接尾辞'     # Verb suffixes
}


class DictionaryManager:
    """Handles JMdict dictionary download, parsing, and lookups."""
    
    def __init__(self):
        self.jmdict_data = None
        self.kanji_to_readings = None
        
    def prepare_dictionary(self) -> bool:
        """Download and extract JMdict if needed, then load it."""
        if not self._download_and_extract_jmdict():
            print("Failed to prepare JMdict dictionary file. Cannot proceed.")
            return False
        
        self.jmdict_data, self.kanji_to_readings = self._load_jmdict()
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
            
    def _load_jmdict(self) -> Tuple[Optional[Dict[str, str]], Optional[Dict[str, List[str]]]]:
        """Parses the JMdict XML file and loads entries into dictionaries."""
        print(f"Loading dictionary from {JMDICT_XML_FILENAME} into memory...")
        jmdict_data = {}
        kanji_to_readings = defaultdict(list)
        
        try:
            # Use iterparse for memory efficiency with large XML files
            context = ET.iterparse(JMDICT_XML_FILENAME, events=('end',))
            _, root = next(context)  # Get the root element

            for event, elem in context:
                if event == 'end' and elem.tag == 'entry':
                    kanji_elements = [k.text for k in elem.findall('k_ele/keb')]
                    reading_elements = [r.text for r in elem.findall('r_ele/reb')]
                    
                    # Map each kanji form to its possible readings
                    for kanji in kanji_elements:
                        for reading in reading_elements:
                            if reading not in kanji_to_readings[kanji]:
                                kanji_to_readings[kanji].append(reading)
                    
                    # Extract definitions (first 2 senses, 2 glosses per sense)
                    definitions = []
                    for sense in elem.findall('sense')[:2]:
                        glosses = sense.findall('gloss')
                        definitions.extend([
                            g.text for g in glosses[:2] 
                            if g.text and g.get('{http://www.w3.org/XML/1998/namespace}lang', 'eng') == 'eng'
                        ])
                        
                    if definitions:
                        definition_str = "; ".join(definitions)
                        # Map all forms to the definition string
                        for form in set(kanji_elements + reading_elements):
                            if form not in jmdict_data:  # Store only the first definition encountered
                                jmdict_data[form] = definition_str
                    
                    # Clear element to save memory
                    elem.clear()
                    
            print(f"Dictionary loaded. Found definitions for {len(jmdict_data)} unique forms.")
            print(f"Created mappings for {len(kanji_to_readings)} kanji forms to readings.")
            return jmdict_data, dict(kanji_to_readings)
            
        except ET.ParseError as e:
            print(f"Error parsing JMdict XML: {e}")
            return None, None
        except Exception as e:
            print(f"Unexpected error during dictionary loading: {e}")
            return None, None
    
    def get_definition(self, lemma: str, word: str) -> str:
        """Looks up English definition in the JMdict data, trying lemma then original word."""
        if not self.jmdict_data:
            return "N/A (Dictionary not loaded)"
        
        # Try lemma first, then surface word if different
        definition = self.jmdict_data.get(lemma)
        if definition:
            return definition
            
        if lemma != word:
            definition = self.jmdict_data.get(word)
            if definition:
                return definition
                
        return "N/A (Not found in JMdict)"
    
    def get_readings(self, lemma: str) -> List[str]:
        """Gets possible readings for a lemma from JMdict."""
        if not self.kanji_to_readings:
            return []
        
        return self.kanji_to_readings.get(lemma, [])
    
    def exists_in_dictionary(self, word: str) -> bool:
        """Checks if a word exists in the JMdict dictionary."""
        if not self.jmdict_data:
            return False
        return word in self.jmdict_data


class JLPTData:
    """Manages JLPT level data loading and lookup."""
    
    def __init__(self):
        self.jlpt_mapping = self._load_jlpt_data()
        
    def _load_jlpt_data(self) -> Dict[str, str]:
        """Loads JLPT level data from CSV files."""
        print("Loading JLPT level data...")
        jlpt_mapping = {}
        
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
    """Handles text processing and reading generation."""
    
    def __init__(self):
        self.kakasi = pykakasi.kakasi()
        # Initialize SudachiPy Tokenizer
        try:
            self.tokenizer = Dictionary().create()
            self.sudachi_mode = SplitMode.A  # More conservative word segmentation
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
                is_separator = (not line or line == "WEBVTT" or 
                                TIMESTAMP_PATTERN.match(line) or line.isdigit())

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

        # Process any remaining parts
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
        # Remove full-width brackets
        text = text.replace('＜', '').replace('＞', '')
        return text
    
    def get_reading(self, text: str) -> str:
        """Generates hiragana reading for Japanese text using pykakasi."""
        result = self.kakasi.convert(text)
        reading = "".join([item.get('hira', '') for item in result]) if result else text
        return reading
    
    def get_context_aware_reading(self, token) -> str:
        """Gets the context-aware reading for a token using SudachiPy."""
        reading = token.reading_form()
        return self._katakana_to_hiragana(reading)
    
    def _katakana_to_hiragana(self, katakana: str) -> str:
        """Converts katakana to hiragana."""
        hiragana_start = ord('ぁ')
        katakana_start = ord('ァ')
        hiragana = []
        
        for char in katakana:
            if 'ァ' <= char <= 'ヶ':
                # Convert by shifting character codes
                hiragana_char = chr(ord(char) - katakana_start + hiragana_start)
                hiragana.append(hiragana_char)
            else:
                # Keep non-katakana characters
                hiragana.append(char)
                
        return ''.join(hiragana)
    
    def is_valid_tokenizer(self) -> bool:
        """Check if the tokenizer is available and valid."""
        return self.tokenizer is not None
        
    def find_best_reading_match(self, context_reading: str, jmdict_readings: List[str]) -> str:
        """Find the closest JMdict reading match to the context-aware reading."""
        if not jmdict_readings:
            return context_reading
        
        # If only one reading, use it
        if len(jmdict_readings) == 1:
            return self._katakana_to_hiragana(jmdict_readings[0])
        
        # Convert all readings to hiragana
        hiragana_readings = [self._katakana_to_hiragana(r) for r in jmdict_readings]
        
        # If context reading matches, use it
        if context_reading in hiragana_readings:
            return context_reading
            
        # Otherwise use the first (most common) reading
        return hiragana_readings[0]
        
    def is_grammatical_element(self, token) -> bool:
        """Determines if a token is a grammatical element rather than a content word."""
        # Check if any part of speech is in the exclusion list
        for pos in token.part_of_speech():
            if pos in EXCLUDE_POS_SUBTYPES:
                return True
                
        # Common auxiliaries and particles to exclude
        surface = token.surface()
        dictionary_form = token.dictionary_form()
        
        aux_endings = {
            'ます', 'です', 'ました', 'でした', 'ない', 'ぬ', 'た', 'だ', 
            'く', 'て', 'で', 'る', 'れる', 'らせる', 'させる', 'ちゃう', 'じゃう',
            'そう', 'よう', 'っぽい', 'がる'
        }
                       
        particles = {
            'は', 'が', 'を', 'に', 'へ', 'と', 'で', 'から', 'まで', 'より',
            'や', 'し', 'ね', 'よ', 'な', 'かな', 'さ', 'わ', 'ぞ', 'ぜ', 'もの',
            'って', 'だけ', 'しか', 'ばかり', 'ほど', 'くらい', 'など', 'なんか'
        }
        
        return (surface in aux_endings or 
                dictionary_form in aux_endings or
                surface in particles or
                dictionary_form in particles)


class VocabularyExtractor:
    """Main class for vocabulary extraction and processing."""
    
    def __init__(self):
        self.dict_manager = DictionaryManager()
        self.jlpt_data = JLPTData()
        self.text_processor = TextProcessor()
        
    def process_vtt_file(self, vtt_path: str, csv_path: str) -> bool:
        """Process a VTT file and extract vocabulary to a CSV file."""
        # Initialize resources
        if not self._initialize_resources():
            return False
            
        # Extract text from VTT file
        print(f"Processing VTT file: {vtt_path}")
        text_lines = self.text_processor.extract_text_from_vtt(vtt_path)
        print(f"Extracted {len(text_lines)} dialogue lines.")
        
        if not text_lines:
            print("No text lines extracted. Exiting.")
            return False
            
        # Extract and process vocabulary
        vocab_items = self._extract_vocabulary(text_lines)
        print(f"Extracted {len(vocab_items)} raw vocabulary items.")
        
        if not vocab_items:
            print("No vocabulary items extracted. Exiting.")
            return False
            
        # Process and filter vocabulary items
        processed_vocab = self._process_vocabulary_items(vocab_items)
        
        # Write to CSV
        return self._write_csv(processed_vocab, csv_path)
    
    def _initialize_resources(self) -> bool:
        """Initialize dictionary and tokenizer."""
        if not self.dict_manager.prepare_dictionary():
            return False
            
        if not self.text_processor.is_valid_tokenizer():
            print("SudachiPy failed to initialize. Cannot proceed.")
            return False
            
        return True
        
    def _extract_vocabulary(self, text_lines: List[str]) -> List[Dict]:
        """Extract vocabulary items from text lines."""
        vocab_items = []
        total_lines = len(text_lines)

        print("Tokenizing sentences and extracting vocabulary...")
        for idx, sentence in enumerate(text_lines):
            print(f"Processing sentence {idx + 1}/{total_lines}", end='\r')
            if not sentence.strip():
                continue
                
            processed_lemmas_in_sentence = set()
            
            try:
                # Tokenize using SudachiPy
                tokens = self.text_processor.tokenizer.tokenize(sentence, self.text_processor.sudachi_mode)
                token_list = list(tokens)  # Convert iterator to list for random access
                
                skip_indices = set()  # Indices to skip (part of compound words)
                
                # Process tokens with compound word detection
                for i in range(len(token_list)):
                    if i in skip_indices:
                        continue
                    
                    # Check for compound words
                    compound_result = self._find_longest_compound_word(token_list, i, sentence)
                    
                    if compound_result:
                        # Process compound word
                        compound_word, num_tokens, compound_info = compound_result
                        
                        # Mark tokens as part of compound
                        for j in range(i, i + num_tokens):
                            skip_indices.add(j)
                        
                        vocab_items.append(compound_info)
                        processed_lemmas_in_sentence.add(compound_info["lemma"])
                    else:
                        # Process single token
                        self._process_single_token(token_list[i], sentence, 
                                                  processed_lemmas_in_sentence, vocab_items)
                        
            except Exception as e:
                print(f"\nError processing sentence: '{sentence}'. Error: {e}")
                continue
                
        print("\nFinished processing sentences.")
        return vocab_items
    
    def _process_single_token(self, token, sentence: str, 
                             processed_lemmas: Set[str], vocab_items: List[Dict]) -> None:
        """Process a single token and add to vocab items if valid."""
        surface_form = token.surface()
        pos = token.part_of_speech()[0]  # Get main POS
        
        # Skip if grammatical element
        if self.text_processor.is_grammatical_element(token):
            return
            
        dictionary_form = token.dictionary_form()

        # Skip if not a target POS, not Japanese, or already processed
        if (pos not in TARGET_POS or 
            not re.search(r'[一-龯ァ-ヶｱ-ﾝﾞﾟ]', dictionary_form) or 
            dictionary_form in processed_lemmas):
            return
        
        # Get reading
        sudachi_reading = self.text_processor.get_context_aware_reading(token)
        jmdict_readings = self.dict_manager.get_readings(dictionary_form)
        
        # Determine the best reading
        if jmdict_readings:
            final_reading = self.text_processor.find_best_reading_match(
                sudachi_reading, jmdict_readings)
        else:
            final_reading = sudachi_reading
        
        # Get definition
        definition = self.dict_manager.get_definition(dictionary_form, surface_form)
        
        # Skip if not in dictionary
        if definition == "N/A (Not found in JMdict)":
            return
        
        # Get JLPT level
        jlpt_level = self.jlpt_data.get_jlpt_level(dictionary_form)
        if jlpt_level == "N/A" and dictionary_form != surface_form:
            jlpt_level = self.jlpt_data.get_jlpt_level(surface_form)
        
        vocab_items.append({
            "lemma": dictionary_form,
            "reading": final_reading,
            "pos": pos,
            "sentence": sentence,
            "definition": definition,
            "jlpt": jlpt_level
        })
        processed_lemmas.add(dictionary_form)
    
    def _find_longest_compound_word(self, tokens, start_idx, sentence):
        """Find the longest valid compound word starting at the given index."""
        if start_idx >= len(tokens):
            return None
            
        max_lookahead = min(5, len(tokens) - start_idx)  # Look ahead up to 5 tokens
        
        # Start with longest possible compound
        for length in range(max_lookahead, 1, -1):
            # Skip if any token is a grammatical element
            if any(self.text_processor.is_grammatical_element(tokens[start_idx + j]) 
                  for j in range(length)):
                continue
            
            # Get surface forms and lemmas
            surface_forms = [tokens[start_idx + j].surface() for j in range(length)]
            lemmas = [tokens[start_idx + j].dictionary_form() for j in range(length)]
            
            # Try different compound combinations
            candidate_compounds = [
                (''.join(surface_forms), ''.join(surface_forms)),  # Surface compound
                (''.join(lemmas), ''.join(surface_forms))          # Lemma compound
            ]
            
            # Keep only unique combinations
            candidate_compounds = list(set(candidate_compounds))
            
            # Check each candidate in dictionary
            for compound_lemma, compound_surface in candidate_compounds:
                if self.dict_manager.exists_in_dictionary(compound_lemma):
                    # Get compound metadata
                    pos = tokens[start_idx].part_of_speech()[0]
                    compound_reading = self.text_processor.get_reading(compound_lemma)
                    
                    # Refine reading if possible
                    jmdict_readings = self.dict_manager.get_readings(compound_lemma)
                    if jmdict_readings:
                        compound_reading = self.text_processor.find_best_reading_match(
                            compound_reading, jmdict_readings)
                    
                    # Get definition and JLPT level
                    definition = self.dict_manager.get_definition(compound_lemma, compound_surface)
                    jlpt_level = self.jlpt_data.get_jlpt_level(compound_lemma)
                    if jlpt_level == "N/A" and compound_lemma != compound_surface:
                        jlpt_level = self.jlpt_data.get_jlpt_level(compound_surface)
                    
                    compound_info = {
                        "lemma": compound_lemma,
                        "reading": compound_reading,
                        "pos": pos,
                        "sentence": sentence,
                        "definition": definition,
                        "jlpt": jlpt_level
                    }
                    
                    return (compound_lemma, length, compound_info)
        
        return None
        
    def _process_vocabulary_items(self, vocab_items: List[Dict]) -> List[Dict]:
        """Process and filter vocabulary items."""
        # Count frequency by lemma and POS
        item_counter = Counter((item['lemma'], item['pos']) for item in vocab_items)

        # Create unique list with frequency counts
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

        # Filter out items without definitions
        filtered_vocab = [item for item in sorted_vocab 
                         if "N/A (Not found in JMdict)" not in item['definition']]
        
        print(f"Filtered out {len(sorted_vocab) - len(filtered_vocab)} items without definitions.")
        print(f"Writing {len(filtered_vocab)} items to CSV.")
        
        return filtered_vocab
        
    def _jlpt_sort_key(self, item: Dict) -> Tuple:
        """Sort key for vocabulary items by JLPT level and frequency."""
        jlpt = item['jlpt']
        
        # Assign sort value to JLPT level
        if jlpt == "N/A":
            jlpt_value = 0  # N/A comes first
        else:
            # Extract numeric part (N1 -> 1, N5 -> 5)
            jlpt_value = int(jlpt[1:])
        
        # Sort by JLPT level, then by descending frequency
        return (jlpt_value, -item['frequency'])
        
    def _write_csv(self, vocab_items: List[Dict], csv_path: str) -> bool:
        """Write vocabulary items to a CSV file."""
        print(f"Writing vocabulary to CSV: {csv_path}")
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'Word (Dictionary Form)', 'Reading', 'Part of Speech', 
                    'English Definition', 'Original Sentence', 'JLPT Level', 'Frequency'
                ]
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
    success = extractor.process_vtt_file(vtt_path, csv_path)
    if success:
        print(f"Successfully extracted vocabulary from {vtt_path} to {csv_path}")
    else:
        print(f"Failed to extract vocabulary from {vtt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract Japanese vocabulary with definitions from a VTT subtitle file."
    )
    parser.add_argument("vtt_file", help="Path to the input VTT file")
    parser.add_argument("csv_file", help="Path to the output CSV file")
    
    args = parser.parse_args()
    
    main(args.vtt_file, args.csv_file) 
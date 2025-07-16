# m_local_agreement.py
import difflib
from typing import List, Optional
import unicodedata

from stream_pipeline.data_package import DataPackage, DataPackageController, DataPackagePhase, DataPackageModule
from stream_pipeline.module_classes import Module, ModuleOptions

import logger
import data

log = logger.get_logger()

class Confirm_Words(Module):
    def __init__(self,
                    offset: float = 0.3,
                    max_confirmed_words: int = 0,
                    transcribe_confirm_if_older_then: float = 2.0,
                    translate_confirm_if_older_then: float = 10.0,
                ) -> None:
        super().__init__(
            ModuleOptions(
                use_mutex=False,
                timeout=5,
            ),
            name="Confirm-Words-Module"
        )
        self.offset = offset
        self.max_confirmed_words = max_confirmed_words
        self.transcribe_confirm_if_older_then = transcribe_confirm_if_older_then  # Confirm words if they are older than this value in seconds
        self.translate_confirm_if_older_then = translate_confirm_if_older_then

        self._confirmed: List[data.Word] = []  # Buffer to store committed words
        self._confirmed_end_time: float = 0.0


    def init_module(self) -> None:
        pass

    def similarity_difflib(self, wort1: str, wort2: str) -> float:
        matcher = difflib.SequenceMatcher(None, wort1, wort2)
        return matcher.ratio()

    def is_similar(self, word1: str, word2: str, max_diff_percentage: float = -1.0) -> bool:
        # Lowercase the words
        word1_l = word1.lower()
        word2_l = word2.lower()

        # Remove symbols and punctuation characters
        def remove_symbols(word: str) -> str:
            # Filter out characters classified as punctuation or symbols
            return ''.join(
                char for char in word 
                if not unicodedata.category(char).startswith(('P', 'S'))
            )

        word1_clean = remove_symbols(word1_l)
        word2_clean = remove_symbols(word2_l)

        if max_diff_percentage == -1.0:
            return word1_clean == word2_clean

        diff = self.similarity_difflib(word1_clean, word2_clean)

        return diff >= max_diff_percentage

        # # Calculate the number of different characters between word1 and word2
        # diff_chars = sum(1 for a, b in zip(word1_clean, word2_clean) if a != b) + abs(len(word1_clean) - len(word2_clean))

        # # Return True if the number of different characters is within the allowed maximum
        # return diff_chars <= max_diff_chars

    def find_word(self, start: float, end: float, words: List[data.Word], offset: float = 0.3) -> Optional[data.Word]:
        for word in words:
            if abs(word.start - start) <= offset and abs(word.end - end) <= offset:
                return word
        return None

    def execute(self, dp: DataPackage[data.AudioData], dpc: DataPackageController, dpp: DataPackagePhase, dpm: DataPackageModule) -> None:
        if not dp.data or dp.data.transcribed_segments is None:
            raise Exception("No transcribed words found")
        if dp.data.audio_buffer_start_after is None:
            raise Exception("No audio buffer start time found")
        if dp.data.audio_buffer_time is None:
            raise Exception("No audio buffer time found")

        audio_buffer_start_after = dp.data.audio_buffer_start_after
        audio_buffer_time = dp.data.audio_buffer_time

        # Collect new words from the transcribed segments
        new_words: List[data.Word] = []
        for segment in dp.data.transcribed_segments:
            if segment.words is None:
                continue
            new_words.extend(segment.words)

        # only_words = [word.word for word in new_words]
        # print(only_words)

        if len(new_words) == 0:
            dp.data.confirmed_words = self._confirmed.copy()
            dp.data.unconfirmed_words = []
            return

        # 1. Split in confirmed, unconfirmed and new words
        new_confirmed: List[data.Word] = []
        new_unconfirmed: List[data.Word] = []
        for new_word in new_words:
            if new_word.start < self._confirmed_end_time - self.offset:
                new_confirmed.append(new_word)
            else:
                new_unconfirmed.append(new_word)

        confirm_if_older_then = 0.0
        if dp.data.task == data.Task.TRANSCRIBE:
            confirm_if_older_then = self.transcribe_confirm_if_older_then
        elif dp.data.task == data.Task.TRANSLATE:
            confirm_if_older_then = self.translate_confirm_if_older_then

        # 2. Check each new_unconfirmed word if it's older than confirm_if_older_then seconds
        words_to_confirm = []
        for new_word in new_unconfirmed:
            if audio_buffer_start_after + audio_buffer_time - new_word.end >= confirm_if_older_then:
                self._confirmed.append(new_word)
                words_to_confirm.append(new_word)

        # Remove confirmed words from unconfirmed list
        # for word in words_to_confirm:
        #     new_unconfirmed.remove(word)

        # Find words which are in new_confirmed and not in confirmed. Use similar
        # time_tolerance = 0.5
        # for new_word in list(reversed(new_confirmed)):
        #     found = False
        #     for confirmed_word in list(reversed(list(self.confirmed))):
        #         if abs(confirmed_word.start - new_word.start) <= time_tolerance and abs(confirmed_word.end - new_word.end) <= time_tolerance:
        #             if self.is_similar(confirmed_word.word, new_word.word):
        #                 found = True

        #                 if confirmed_word.word != new_word.word and confirmed_word.probability - 0.1 < new_word.probability:
        #                     # print(f"Word changed: {confirmed_word.word} -> {new_word.word}")
        #                     confirmed_word.word = new_word.word
        #                     confirmed_word.start = new_word.start
        #                     confirmed_word.end = new_word.end
        #                     confirmed_word.probability = new_word.probability

        #                 break
        #     # if not found:
        #     #     # if word isnt older then 10s
        #     #     if new_word.end - audio_buffer_start_after > 20:
        #     #         self.confirmed.append(new_word)

        # Remove words from confirmed which are not confidant enough < 0.2
        self._confirmed = [word for word in self._confirmed if word.probability >= 0.2]

        # sort confirmed words by start time
        self._confirmed = sorted(self._confirmed, key=lambda x: x.start)

        # remove words which times are overlapping.
        to_remove_list = []
        for i in range(len(self._confirmed) - 1):
            if self._confirmed[i].end > self._confirmed[i + 1].start:
                if self.is_similar(self._confirmed[i].word, self._confirmed[i + 1].word, 0.7):
                    to_remove_list.append(i)
                    i = i + 1

        while len(to_remove_list) > 0:
            i = to_remove_list.pop(0)
            self._confirmed.pop(i)
            to_remove_list = [x-1 for x in to_remove_list]


        # Ensure that the number of confirmed words does not exceed the max_confirmed_words limit
        if len(self._confirmed) > self.max_confirmed_words:
            self._confirmed = self._confirmed[-self.max_confirmed_words:]

        if len(self._confirmed) > 0:
            self._confirmed_end_time = self._confirmed[-1].end

        if len(self._confirmed) > self.max_confirmed_words:
            self._confirmed = self._confirmed[-self.max_confirmed_words:]

        # Update data package confirmed and unconfirmed words
        # only_words = [word.word for word in self.confirmed]
        # print(only_words)
        dp.data.confirmed_words = self._confirmed.copy()
        dp.data.unconfirmed_words = new_unconfirmed

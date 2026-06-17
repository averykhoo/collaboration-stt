from typing import List, Tuple, Optional
import sys
import codecs

def my_reader(in_path):
    fid = codecs.open(in_path,'r','utf-8-sig')
    return fid

def my_appender(out_path):
    fid = codecs.open(out_path,'a','utf-8')
    return fid

def my_writer(out_path):
    fid = codecs.open(out_path,'w','utf-8')
    return fid

def write_line(fid, str):
    fid.write(str + "\n")

def continuous_replace(src,dest,inString):
    prevString = inString

    while True:
        inString = inString.replace(src,dest)
        if inString == prevString:
            break
        prevString = inString

    return inString


class AccuracyStatistics:
    """
    An accumulator for accuracy statistics, for a set of pair of transcribed texts and their corresponding
    reference texts.
    """

    def __init__(self,
                 allowed_replacements: List[Tuple[str, str]],
                 allowed_insertions: List[str],
                 allowed_deletions: List[str]):
        """
        Constructor.

        :param allowed_replacements: A list of pairs of words or texts that are allowed to be substituted.
        :param allowed_insertions: A list of pairs of words or texts that are allowed to be inserted to the transcription.
        :param allowed_deletions: A list of pairs of words or texts that are allowed to be deleted in the transcription.
        """
        # Copy the allowed substitutions, insertions and deletions.
        self._allowed_replacements = []
        for first, second in allowed_replacements:
            self._allowed_replacements.append((first, second))
            self._allowed_replacements.append((second, first))

        self._allowed_insertions = []
        for text in allowed_insertions:
            self._allowed_insertions.append(text)

        self._allowed_deletions = []
        for text in allowed_deletions:
            self._allowed_deletions.append(text)

        # Reset the error counters.
        self._num_matches = 0
        self._num_substitutions = 0
        self._num_insertions = 0
        self._num_deletions = 0

        self._substitution_counter = {}
        self._insertion_counter = {}
        self._deletion_counter = {}

        # Initialize the work matrices for the alignment computations.
        self._mat_dim = 0
        self._scores_mat = None
        self._ops_mat = None

    def matches(self) -> int:
        """
        Get the total number of word matches in all aligned sequences.

        :return: The total number of word matches.
        """
        return self._num_matches

    def substitutions(self) -> int:
        """
        Get the total number of word substitutions in all aligned sequences.

        :return: The total number of word substitutions.
        """
        return self._num_substitutions

    def insertions(self) -> int:
        """
        Get the total number of inserted word in all aligned sequences.

        :return: The total number of word insertions.
        """
        return self._num_insertions

    def deletions(self) -> int:
        """
        Get the total number of deleted word in all aligned sequences.

        :return: The total number of word deletions.
        """
        return self._num_deletions

    def word_error_rate(self) -> float:
        """
        Get the overall word error rate of all transcriptions.

        :return: The word error rate.
        """
        denominator = self._num_matches + self._num_substitutions + self._num_deletions
        if denominator == 0:
            return 0

        numerator = self._num_substitutions + self._num_insertions + self._num_deletions
        return numerator / denominator

    def recall(self) -> float:
        """
        Get the overall recall of all transcriptions.

        :return: The recall (between 0 and 1).
        """
        if self._num_matches == 0:
            return 0

        denominator = self._num_matches + self._num_substitutions + self._num_deletions
        return self._num_matches / denominator

    def precision(self) -> float:
        """
        Get the overall precision of all transcriptions.

        :return: The precision (between 0 and 1).
        """
        if self._num_matches == 0:
            return 0

        denominator = self._num_matches + self._num_substitutions + self._num_insertions
        return self._num_matches / denominator

    def f1_score(self) -> float:
        """
        Get the overall F1 score of all transcriptions.

        :return: The F1 score (between 0 and 1).
        """
        if self._num_matches == 0:
            return 0

        rec = self.recall()
        prec = self.precision()
        return (2 * rec * prec) / (rec + prec)

    def substitution_counters(self) -> List[Tuple[str, str, int]]:
        """
        Get all substitutions, sorted from the most common ones to the least common ones.

        :return: A list of tuples of (reference text, transcribed text, number of occurrences)
                 for each unique substitution.
        """
        return [(texts[0], texts[1], count) for texts, count in sorted(self._substitution_counter.items(),
                                                                       key=lambda pair: (-pair[1], pair[0]))]

    def insertion_counters(self) -> List[Tuple[str, int]]:
        """
        Get all insertions, sorted from the most common ones to the least common ones.

        :return: A list of tuples of (inserted transcribed text, number of occurrences)
                 for each unique insertion.
        """
        return [(text, count) for text, count in sorted(self._insertion_counter.items(),
                                                        key=lambda pair: (-pair[1], pair[0]))]

    def deletion_counters(self) -> List[Tuple[str, int]]:
        """
        Get all deletions, sorted from the most common ones to the least common ones.

        :return: A list of tuples of (deleted transcribed text, number of occurrences)
                 for each unique deletion.
        """
        return [(text, count) for text, count in sorted(self._deletion_counter.items(),
                                                        key=lambda pair: (-pair[1], pair[0]))]

    def accumulate(self,
                   reference_words: List[str],
                   transcription_words: List[str]):
        """
        Accumulate statistics for the current transcription, given the corresponding reference text.

        :param reference_words: The list of words that comprise the reference text.
        :param transcription_words: The list of words that comprise the transcribed text.
        """
        # Get a list of aligned word pairs:
        aligned_pairs = self.__global_alignment(reference_words,
                                                transcription_words)

        # Analyze the alignment.
        aligned_len = len(aligned_pairs)
        start_ind = 0

        while start_ind < aligned_len:
            # Find the next pair of matching words from the current start index (inclusive).
            next_match_ind = start_ind
            while next_match_ind < aligned_len:
                ref_word, trans_word = aligned_pairs[next_match_ind]
                if (not ref_word is None) and (not trans_word is None) and ref_word == trans_word:
                    break

                next_match_ind += 1

            if next_match_ind == start_ind:
                self._num_matches += 1
                start_ind += 1
                continue

            # Go over the range of mismatched pairs.
            mismatched_ref_words = []
            mismatched_trans_words = []
            n_subs = 0
            n_inss = 0
            n_dels = 0

            for ref_word, trans_word in aligned_pairs[start_ind: next_match_ind]:
                if ref_word is None:
                    # We have an insertion of the transcription word.
                    mismatched_trans_words.append(trans_word)
                    n_inss += 1

                elif trans_word is None:
                    # We have a deletion of the reference word.
                    mismatched_ref_words.append(ref_word)
                    n_dels += 1

                else:
                    # We have a substitution.
                    mismatched_ref_words.append(ref_word)
                    mismatched_trans_words.append(trans_word)
                    n_subs += 1

            if len(mismatched_ref_words) == 0:
                # We have a sequence of insertions (possibly of length 1):
                # Check if the sequence of transcription words make up an allowed insertion.
                inserted_text = ' '.join(mismatched_trans_words)
                if not inserted_text in self._allowed_insertions:
                    # Count the insertion if it is not allowed.
                    if inserted_text in self._insertion_counter:
                        self._insertion_counter[inserted_text] += 1
                    else:
                        self._insertion_counter[inserted_text] = 1

                    self._num_insertions += n_inss

            elif len(mismatched_trans_words) == 0:
                # We have a sequence of deletions (possibly of length 1):
                # Check if the sequence of reference words make up an allowed deletion.
                deleted_text = ' '.join(mismatched_ref_words)

                if not deleted_text in self._allowed_deletions:
                    # Count the deletion if it is not allowed.
                    if deleted_text in self._deletion_counter:
                        self._deletion_counter[deleted_text] += 1
                    else:
                        self._deletion_counter[deleted_text] = 1

                    self._num_deletions += n_dels

            else:
                # We have a substitution:
                # Check if the reference text and the transcribed text correspond to an allowed substitution.
                ref_text = ' '.join(mismatched_ref_words)
                trans_text = ' '.join(mismatched_trans_words)
                key_pair = (ref_text, trans_text)

                if not key_pair in self._allowed_replacements:
                    if key_pair in self._substitution_counter:
                        self._substitution_counter[key_pair] += 1
                    else:
                        self._substitution_counter[key_pair] = 1

                    # Accumulate all errors counted for the current pair:
                    self._num_substitutions += n_subs
                    self._num_insertions += n_inss
                    self._num_deletions += n_dels

                else:
                    # Count the allowed substitution as a single match.
                    self._num_matches += 1

            # Proceed to the next word pair after the mismatching sequence.
            start_ind = next_match_ind

    def __global_alignment(self,
                           reference_words: List[str],
                           transcription_words: List[str]) -> List[Tuple[Optional[str], Optional[str]]]:
        # Define some internal constants.
        op_null = 0
        op_pair = 1
        op_insertion = 2
        op_deletion = 3

        match_weight = 2.0
        substitution_weight = -1.0
        insertion_weight = -0.75
        deletion_weight = -0.75

        # Resize the work matrices, if necessary.
        num_ref_words = len(reference_words)
        num_trans_words = len(transcription_words)

        if max(num_ref_words, num_trans_words) >= self._mat_dim:
            self._mat_dim = 1 + max(num_ref_words, num_trans_words)

            self._scores_mat = []
            self._ops_mat = []
            for i in range(self._mat_dim):
                self._scores_mat.append([0.0] * self._mat_dim)
                self._ops_mat.append([0] * self._mat_dim)

        # Fill the first row of the matrices, which corresponds to a sequence of insertions.
        self._scores_mat[0][0] = 0.0
        self._ops_mat[0][0] = op_null

        for j in range(num_trans_words):
            self._scores_mat[0][j + 1] = self._scores_mat[0][j] + insertion_weight
            self._ops_mat[0][j + 1] = op_insertion

        # Fill the rest of the rows.
        for i in range(num_ref_words):
            # The first entry in each row corresponds to a sequence of deletions.
            self._scores_mat[i + 1][0] = self._scores_mat[i][0] + deletion_weight

            # Fill the rest of the row entries.
            for j in range(num_trans_words):
                # Consider the pair operation (match or substitution) between the i'th reference word
                # and the j'th transcription word.
                if reference_words[i] == transcription_words[j]:
                    max_score = self._scores_mat[i][j] + match_weight
                else:
                    max_score = self._scores_mat[i][j] + substitution_weight
                best_op = op_pair

                # Consider the deletion of the current reference word.
                deletion_score = self._scores_mat[i][j + 1] + deletion_weight
                if deletion_score > max_score:
                    max_score = deletion_score
                    best_op = op_deletion

                # Consider the insertion of the current transcription word.
                insertion_score = self._scores_mat[i + 1][j] + insertion_weight
                if insertion_score > max_score:
                    max_score = insertion_score
                    best_op = op_insertion

                # Fill the optimal option in the matrices.
                self._scores_mat[i + 1][j + 1] = max_score
                self._ops_mat[i + 1][j + 1] = best_op

        # Trace back the optimal alignment.
        aligned_pairs = []
        curr_i = num_ref_words
        curr_j = num_trans_words

        while curr_i > 0 or curr_j > 0:
            op_code = self._ops_mat[curr_i][curr_j]

            if op_code == op_pair:
                # Pair operation.
                aligned_pairs.append((reference_words[curr_i - 1], transcription_words[curr_j - 1]))
                curr_i -= 1
                curr_j -= 1

            elif op_code == op_insertion:
                # Insertion operation.
                aligned_pairs.append((None, transcription_words[curr_j - 1]))
                curr_j -= 1

            elif op_code == op_deletion:
                # Deletion operation.
                aligned_pairs.append((reference_words[curr_i - 1], None))
                curr_i -= 1

            else:
                break

        # Reverse the list of aligned word pairs, as we have constructed it from end to beginning.
        aligned_pairs.reverse()
        return aligned_pairs


def calculate_wer(ground_truth_file, predicted_arr, allowed_replacements_file):
    """
    Calculate WER along with recall, precision and F1-score.
    This function prints all relevant information of the screen.
    See more information in main function.

    :param ground_truth_file: The text file containing ground truth info.
    :param predicted_arr: array of predicted words

    :return AccuracyStatistics class.
    """
    
    # Parse groun truth file.
    r_ref = my_reader(ground_truth_file)
    ref_list = []
    for line_ref in r_ref:
        line_ref = line_ref.strip()
        if not line_ref:
            continue
        line_ref = line_ref.replace("<unk>","")
        line_ref = continuous_replace("  "," ",line_ref)
        words = line_ref.split()
        ref_list.extend(words)

    # Parse allower replacement file.
    r = my_reader(allowed_replacements_file)
    allowed_r = []
    for line in r:
        line = line.strip()
        if not line:
            continue
        a = line.split(",")
        allowed_r.append((a[0],a[1]))
    r.close()

    # Initiate AccuracyStatistics class
    stats = AccuracyStatistics(allowed_replacements=allowed_r,
                       allowed_insertions=[],
                       allowed_deletions=[])

    # Calculate WER related statistics
    stats.accumulate(ref_list, predicted_arr)

    # Write output statistics    
    print('M = %d, S = %d, I = %d, D = %d' % (stats._num_matches, stats._num_substitutions, stats._num_insertions, stats._num_deletions))
    print('WER = %.1f%%, recall = %.1f%%, precision = %.1f%%, F1-score = %.1f%%' % (100 * stats.word_error_rate(), 100 * stats.recall(), 100 * stats.precision(), 100 * stats.f1_score()))

    print('Substitutions:')
    for ref, trans, count in stats.substitution_counters():
        print('    %s  <->  %s  : %d occurrence(s)' % (ref, trans, count))

    print('Insertions:')
    for trans, count in stats.insertion_counters():
        print('    %s  : %d occurrence(s)' % (trans, count))

    print('Deletions:')
    for ref, count in stats.deletion_counters():
        print('    %s  : %d occurrence(s)' % (ref, count))

    return stats

if __name__ == '__main__':

    r_ref = my_reader(sys.argv[1])
    r_obs = my_reader(sys.argv[2])

    replacement_file = sys.argv[3]
    insertion_file   = sys.argv[4]
    deletion_file    = sys.argv[5]
    out_stat_file    = sys.argv[6]

    ref_list = []
    obs_list = []

    # Read reference
    for line_ref in r_ref:
        line_ref = line_ref.strip()
        if not line_ref:
            continue
        line_ref = line_ref.replace("<unk>", "")
        line_ref = continuous_replace("  ", " ", line_ref)
        ref_list.extend(line_ref.split())

    # Read hypothesis
    for line_obs in r_obs:
        line_obs = line_obs.strip()
        if not line_obs:
            continue
        line_obs = line_obs.replace("<unk>", "")
        line_obs = continuous_replace("  ", " ", line_obs)
        obs_list.extend(line_obs.split())

    # Read allowed replacements
    allowed_r = []
    r = my_reader(replacement_file)
    for line in r:
        line = line.strip()
        if not line:
            continue
        a = line.split(",", 1)
        allowed_r.append((a[0], a[1]))
    r.close()

    # Read allowed insertions
    allowed_i = []
    r = my_reader(insertion_file)
    for line in r:
        line = line.strip()
        if not line:
            continue
        allowed_i.append(line)
    r.close()

    # Read allowed deletions
    allowed_d = []
    r = my_reader(deletion_file)
    for line in r:
        line = line.strip()
        if not line:
            continue
        allowed_d.append(line)
    r.close()

    # Initialize statistics
    stats = AccuracyStatistics(
        allowed_replacements=allowed_r,
        allowed_insertions=allowed_i,
        allowed_deletions=allowed_d
    )

    # Accumulate statistics
    stats.accumulate(ref_list, obs_list)

    # Write output
    w = my_writer(out_stat_file)

    write_line(w, 'M = %d, S = %d, I = %d, D = %d'
               % (stats._num_matches,
                  stats._num_substitutions,
                  stats._num_insertions,
                  stats._num_deletions))

    write_line(w, 'WER = %.1f%%, recall = %.1f%%, precision = %.1f%%, F1-score = %.1f%%'
               % (100 * stats.word_error_rate(),
                  100 * stats.recall(),
                  100 * stats.precision(),
                  100 * stats.f1_score()))

    write_line(w, 'Substitutions:')
    for ref, trans, count in stats.substitution_counters():
        write_line(w, '    %s  <->  %s  : %d occurrence(s)' % (ref, trans, count))

    write_line(w, 'Insertions:')
    for trans, count in stats.insertion_counters():
        write_line(w, '    %s  : %d occurrence(s)' % (trans, count))

    write_line(w, 'Deletions:')
    for ref, count in stats.deletion_counters():
        write_line(w, '    %s  : %d occurrence(s)' % (ref, count))

    w.close()




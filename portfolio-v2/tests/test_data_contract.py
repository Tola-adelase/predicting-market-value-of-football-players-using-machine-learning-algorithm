"""Meaningful input/split checks; test data is synthetic and never used as results."""
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import pandas as pd
from valuation import load_data, split_players, assert_disjoint, PROXIES

class DataContractTests(unittest.TestCase):
    def test_players_never_cross_splits_and_rows_are_preserved(self):
        df = pd.DataFrame({'_player_group': [str(i // 2) for i in range(200)]})
        parts = split_players(df)
        self.assertEqual([len(p) for p in parts], [120, 40, 40])
        self.assertEqual(sum(len(p) for p in parts), len(df))
        self.assertEqual([p.index.tolist() for p in parts],
                         [p.index.tolist() for p in split_players(df)])
        with self.assertRaises(AssertionError):
            assert_disjoint([parts[0], parts[0], parts[2]])

    def test_bad_targets_removed_and_missing_predictors_retained(self):
        df = pd.DataFrame({c: [1.] * 103 for c in ['Value'] + PROXIES})
        df['Name'] = ['Player ' + str(i) for i in range(103)]
        df.loc[0, 'Value'] = -1
        df.loc[1, 'Value'] = float('inf')
        df.loc[2, 'Name'] = ''
        df.loc[3, 'Age'] = None
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'players.csv'; df.to_csv(p, index=False)
            clean, audit = load_data(p)
        self.assertEqual(len(clean), 100)
        self.assertEqual(audit['invalid_target_or_identity_removed'], 3)
        self.assertEqual(clean['Age'].isna().sum(), 1)
        self.assertEqual(len(audit['source_sha256']), 64)

    def test_missing_data_stops_without_substitution(self):
        with self.assertRaises(FileNotFoundError):
            load_data('/missing-original-data/players.csv')

if __name__ == '__main__':
    unittest.main()

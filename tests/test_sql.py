import unittest
import tempfile
import os.path
import shutil

import numpy
import datafusion

# used to write parquet files
import pyarrow.parquet


def _write_parquet(path):
    a = numpy.concatenate([numpy.random.normal(0, 0.01, size=50), numpy.random.normal(50, 0.01, size=50)])

    table = pyarrow.Table.from_arrays([pyarrow.array(a)], names=['a'])
    pyarrow.parquet.write_table(table, path)
    return path


class TestCase(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_basics(self):
        ctx = datafusion.ExecutionContext()
        with self.assertRaises(Exception):
            datafusion.Context().sql("SELECT a FROM b", 10)

    def test_register(self):
        ctx = datafusion.ExecutionContext()
        ctx.register_parquet("t", "../arrow/testing/data/parquet/generated_simple_numerics/blogs.parquet")

        self.assertEqual(ctx.tables(), {"t"})

    def test_execute(self):
        ctx = datafusion.ExecutionContext()

        # single column, "a"
        path = _write_parquet(os.path.join(self.test_dir, 'a.parquet'))
        ctx.register_parquet("t", path)

        self.assertEqual(ctx.tables(), {"t"})

        # count
        expected = {'COUNT': numpy.array([100])}
        self.assertEqual(expected, ctx.sql("SELECT COUNT(a) FROM t", 20))

        # where
        expected = {'COUNT': numpy.array([50])}
        self.assertEqual(expected, ctx.sql("SELECT COUNT(a) FROM t WHERE a > 10", 20))

        # group by
        result = ctx.sql("SELECT (a > 50), COUNT(a) FROM t GROUP BY CAST((a > 10.0) AS int)", 20)
        expected = {'CAST': numpy.array([0, 1]), 'COUNT': numpy.array([50, 50])}
        numpy.testing.assert_equal(expected, result)

        # order by
        result = ctx.sql("SELECT a, CAST(a AS int) FROM t ORDER BY a DESC LIMIT 2", 20)['CAST']
        expected = numpy.array([50, 50])
        numpy.testing.assert_equal(expected, result)

    def test_cast(self):
        """
        Verify that we can run
        """
        ctx = datafusion.ExecutionContext()

        path = _write_parquet(os.path.join(self.test_dir, 'a.parquet'))
        ctx.register_parquet("t", path)

        valid_types = [
            'smallint',
            'int',
            'bigint',
            'float(32)',
            'float(64)',
            'float',
        ]

        select = ', '.join([f'CAST(9 AS {t})' for t in valid_types])

        # can execute, which implies that we can cast
        ctx.sql(f'SELECT {select} FROM t', 20)

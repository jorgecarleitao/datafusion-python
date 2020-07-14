import unittest
import tempfile
import os.path
import shutil

import numpy
import datafusion

# used to write parquet files
import pyarrow.parquet


def _data():
    return numpy.concatenate([numpy.random.normal(0, 0.01, size=50), numpy.random.normal(50, 0.01, size=50)])


def _data_with_nulls():
    data = numpy.random.normal(0, 0.01, size=50)
    mask = numpy.random.randint(0, 2, size=50)
    data[mask==0] = numpy.NaN
    return data

def _data_string():
    data = numpy.random.choice(["aaa" , "b"], size=5)
    data = numpy.array(data, dtype='O')
    mask = numpy.random.randint(0, 2, size=5)
    data[mask==0] = None
    return data

def _write_parquet(path, data):
    table = pyarrow.Table.from_arrays([pyarrow.array(data)], names=['a'])
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
        path = _write_parquet(os.path.join(self.test_dir, 'a.parquet'), _data())
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

        path = _write_parquet(os.path.join(self.test_dir, 'a.parquet'), _data())
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

    def test_udf(self):
        data = _data()

        ctx = datafusion.ExecutionContext()

        ctx.register_udf("iden", lambda x: abs(x), ['float64'], 'float64')

        # write to disk
        path = _write_parquet(os.path.join(self.test_dir, 'a.parquet'), data)
        ctx.register_parquet("t", path)

        result = ctx.sql("SELECT iden(a) AS tt FROM t", 20)

        # compute the same operation here
        expected = {'tt': numpy.abs(data)}

        numpy.testing.assert_equal(expected, result)

    def test_nulls(self):
        data = _data_with_nulls()

        ctx = datafusion.ExecutionContext()

        # write to disk
        path = _write_parquet(os.path.join(self.test_dir, 'a.parquet'), data)
        ctx.register_parquet("t", path)

        result = ctx.sql("SELECT a AS tt FROM t", 20)

        numpy.testing.assert_equal(data, result['tt'])

    def test_nulls_udf(self):
        data = _data_with_nulls()

        ctx = datafusion.ExecutionContext()

        ctx.register_udf("iden", lambda x: abs(x), ['float64'], 'float64')

        # write to disk
        path = _write_parquet(os.path.join(self.test_dir, 'a.parquet'), data)
        ctx.register_parquet("t", path)

        result = ctx.sql("SELECT iden(a) AS tt FROM t", 20)

        # compute the same operation here
        expected = {'tt': numpy.abs(data)}

        numpy.testing.assert_equal(expected, result)

    def test_strings(self):
        data = _data_string()

        ctx = datafusion.ExecutionContext()

        # write to disk
        path = _write_parquet(os.path.join(self.test_dir, 'a.parquet'), data)
        ctx.register_parquet("t", path)

        result = ctx.sql("SELECT a AS tt FROM t", 20)

        # known issue: null values are converted to ''
        data[data==None] = ''

        numpy.testing.assert_equal(data, result['tt'])

import unittest
import tempfile
import os.path
import shutil

import numpy
import datafusion
import pyarrow.parquet


def _write_parquet(path):
    a = numpy.random.normal(100, size=100)

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

        path = _write_parquet(os.path.join(self.test_dir, 'a.parquet'))

        ctx.register_parquet("t", path)

        self.assertEqual(ctx.tables(), {"t"})

        # execution works; result is wrong :P
        self.assertEqual(5, ctx.sql("SELECT a FROM t", 20))

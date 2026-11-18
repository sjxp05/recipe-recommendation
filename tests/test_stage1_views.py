import json
import re
import unittest

from training.stage1_views import (drop_raw_and_other, masked_assembly,
                                   prune_null_fields, unified_assembly)


class Stage1ViewsTests(unittest.TestCase):
    def test_drop_raw_and_other_keeps_remaining_operations_in_order(self):
        source = ('RECIPE {"name":"soup"}\n'
                  'ING I0 {"source":"onion","unified_id":10}\n'
                  'OP {"verb":"chop","other":"with a knife",'
                  '"clauses":[{"marker":"then","other":"finely","time":0}]}\n'
                  'RAW {"raw":"unparsed instruction"}\n'
                  'OP {"verb":"boil","time":10}\n')
        result = drop_raw_and_other(source)
        self.assertEqual([line.split(" ", 1)[0] for line in result.splitlines()],
                         ["RECIPE", "ING", "OP", "OP"])
        self.assertEqual(json.loads(result.splitlines()[2][3:]),
                         {"verb": "chop", "clauses": [{"marker": "then", "time": 0}]})
        self.assertEqual(json.loads(result.splitlines()[3][3:]),
                         {"verb": "boil", "time": 10})

    def test_prune_null_fields_preserves_operation_order_and_values(self):
        source = ('RECIPE {"name":"soup"}\n'
                  'ING I0 {"source":"onion","unified_id":null}\n'
                  'OP {"verb":"chop","object":"@I0","time":null,"clauses":'
                  '[{"marker":"then","temperature":null,"value":0}]}\n'
                  'OP {"verb":"boil","time":10,"optional":false}\n')
        result = prune_null_fields(source)
        lines = result.splitlines()
        self.assertEqual([line.split(" ", 1)[0] for line in lines],
                         ["RECIPE", "ING", "OP", "OP"])
        self.assertEqual(json.loads(lines[1].split(" ", 2)[2]), {"source": "onion"})
        self.assertEqual(json.loads(lines[2][3:]),
                         {"verb": "chop", "object": "@I0",
                          "clauses": [{"marker": "then", "value": 0}]})
        self.assertEqual(json.loads(lines[3][3:]),
                         {"verb": "boil", "time": 10, "optional": False})

    def test_unified_assembly_replaces_legacy_ids(self):
        source = ('RECIPE {"name":"pork soup"}\n'
                  'ING I0 {"source":"lean pork chops","id":4207,'
                  '"general_id":4207,"name":"lean pork chops","status":"id_normalized"}\n'
                  '\nRAW {"raw":"cook @I0"}\n')
        mapping = ({(4207, "lean pork chops"): 9001}, {"lean pork chops": {9001}})
        result = unified_assembly(source, mapping)
        ingredient = json.loads(result.splitlines()[1].split(" ", 2)[2])
        self.assertEqual(json.loads(result.splitlines()[0][7:]), {"name": "pork soup"})
        self.assertEqual(ingredient["unified_id"], 9001)
        self.assertNotIn("id", ingredient)
        self.assertNotIn("general_id", ingredient)

    def test_masked_assembly_removes_ingredient_identity(self):
        source = ('RECIPE {"name":"lean pork chops"}\n'
                  'ING I0 {"source":"lean pork chops","name":"lean pork chops",'
                  '"unified_id":9001}\n'
                  '\nRAW {"raw":"cook lean pork chops and @I0",'
                  '"mentions":[{"name":"lean pork chops","id":4207}]}\n')
        tag = ("I0", "육류", "요리 중심재료", 0.910849, "[MASK:I0|육류|요리 중심재료|0.910849]")

        def redact(value, patterns, tags):
            return re.sub("lean pork chops|@I0", tags["I0"], value)

        result = masked_assembly(source, [tag], None, redact)
        self.assertIn('ING I0 ["육류", "요리 중심재료", 0.910849]', result)
        self.assertNotIn("lean pork chops", result)
        self.assertNotIn("9001", result)
        self.assertNotIn("4207", result)
        self.assertNotIn("mentions", result)


if __name__ == "__main__":
    unittest.main()

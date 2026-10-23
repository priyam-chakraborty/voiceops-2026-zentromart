import unittest
import run


class UnsupportedMidnightDeliveryTests(unittest.TestCase):
    def bonus(self, question, section):
        return run._retrieval_topic_bonus(question, {"section_id": section})

    def test_unsupported_midnight_schedule_gets_no_topic_bonus(self):
        q = "Can I schedule a delivery at midnight?"
        for section in ("DEL-1", "DEL-3", "DEL-4", "DEL-5"):
            with self.subTest(section=section):
                self.assertEqual(self.bonus(q, section), 0.0)

    def test_supported_delivery_area_query_keeps_bonus(self):
        self.assertEqual(self.bonus("Can you deliver to Dubai?", "DEL-5"), 0.115)

    def test_supported_missed_delivery_query_keeps_bonus(self):
        self.assertEqual(
            self.bonus("What happens if I'm not home when the courier comes?", "DEL-3"),
            0.095,
        )

    def test_general_delivery_times_still_retrieve_delivery_policy(self):
        self.assertEqual(self.bonus("What are the standard delivery times?", "DEL-1"), 0.045)


if __name__ == "__main__":
    unittest.main(verbosity=2)

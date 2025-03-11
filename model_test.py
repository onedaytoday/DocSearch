import unittest

import torch

from LLM2VecModel import LLM2VecModel


class MyTestCase(unittest.TestCase):
    HF_PASSCODE = "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"
    MODEL_ID = "McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp"
    MODEL_SECONDARY_ID = "McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised"



    model = LLM2VecModel(MODEL_ID, MODEL_SECONDARY_ID, token=HF_PASSCODE)

    def test_model1(self):
        l2v = self.model
        instruction = (
            "Given a search query, retrieve relevant information, keywords, or topics that answers the query:"
        )

        # List of queries to search for relevant information, keywords, or topics
        queries = [
            [instruction, "how much protein should a female eat"],
            [instruction, "summit define"],
            [instruction, "best cars in the market today"],
            [instruction, "being poor"],

        ]

        q_reps = l2v.encode(queries)

        # Encoding documents. Instruction are not required for documents
        documents = [
            "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per "
            "day. But, as you can see from this chart, you'll need to increase that if you're expecting or training "
            "for a marathon. Check out the chart below to see how much protein you should be eating each day.",
            "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a "
            "mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or "
            "more governments.",
            "Luxury cars are high-end vehicles designed with superior craftsmanship, advanced technology, "
            "and exceptional comfort to offer an elevated driving experience. They often feature premium materials, "
            "cutting-edge performance, and exclusive designs, catering to a sophisticated and affluent clientele.",
        ]
        d_reps = l2v.encode(documents)

        # Compute cosine similarity
        q_reps_norm = torch.nn.functional.normalize(q_reps, p=2, dim=1)
        d_reps_norm = torch.nn.functional.normalize(d_reps, p=2, dim=1)
        cos_sim = torch.mm(q_reps_norm, d_reps_norm.transpose(0, 1))

        print(cos_sim)

    def test_model2(self):
        l2v = self.model
        # List of queries to search for relevant information, keywords, or topics
        queries = [
            "how to set up Raspberry Pi",  # Related to document[0]
            "how to install Raspberry Pi OS",  # Related to document[1]
            "how to connect Raspberry Pi to a monitor",  # Related to document[2]
            "Raspberry Pi GPIO pinout",  # Related to document[3]
            "how to program on Raspberry Pi",  # Related to document[4]
            "how to use a Raspberry Pi camera",  # Related to document[5]
            "what is the best power supply for Raspberry Pi",  # Related to document[6]
            "how to use Raspberry Pi for home automation",  # Related to document[7]
            "how to connect Raspberry Pi to Wi-Fi",  # Related to document[8]
            "how to troubleshoot Raspberry Pi performance issues"  # Related to document[9]
        ]

        # List of intended correct document indices for each query
        intended_indices = [
            0,  # Query: "how to set up Raspberry Pi"
            1,  # Query: "how to install Raspberry Pi OS"
            2,  # Query: "how to connect Raspberry Pi to a monitor"
            3,  # Query: "Raspberry Pi GPIO pinout"
            4,  # Query: "how to program on Raspberry Pi"
            5,  # Query: "how to use a Raspberry Pi camera"
            6,  # Query: "what is the best power supply for Raspberry Pi"
            7,  # Query: "how to use Raspberry Pi for home automation"
            8,  # Query: "how to connect Raspberry Pi to Wi-Fi"
            9  # Query: "how to troubleshoot Raspberry Pi performance issues"
        ]

        q_reps = l2v.encode(queries)

        # Encoding documents
        documents = [
            "Setting up your Raspberry Pi is easy. First, download Raspberry Pi OS from the official website and use "
            "an imaging tool like Balena Etcher to write the OS image onto a microSD card. Once written, insert the "
            "card into the Raspberry Pi, connect the monitor, keyboard, and mouse, and power it up.",
            "To install Raspberry Pi OS, begin by downloading the Raspberry Pi Imager. Select the Raspberry Pi OS "
            "from the menu and choose the storage device (SD card). Once the installation is complete, insert the "
            "card into the Raspberry Pi and boot it up.",
            "Connecting your Raspberry Pi to a monitor requires an HDMI cable. Depending on the model of Raspberry Pi "
            "you have, use either the standard HDMI or micro-HDMI port to connect to your display.",
            "The GPIO (General Purpose Input/Output) pins on the Raspberry Pi allow you to interface with external "
            "hardware. There are 40 pins on the Raspberry Pi 4, which are split into power, ground, and GPIO signal "
            "pins. Refer to the pinout diagram for exact pin usage.",
            "Programming on the Raspberry Pi can be done in various languages such as Python, Java, or C++. The most "
            "commonly used language is Python due to its simplicity and compatibility with most Pi projects.",
            "Using the Raspberry Pi camera module involves connecting the camera to the dedicated camera serial "
            "interface (CSI) port. Once connected, you can use the `raspistill` and `raspivid` commands to capture "
            "images and videos.",
            "For the Raspberry Pi, the recommended power supply is 5V, 3A with a micro-USB or USB-C connector, "
            "depending on the model. A stable power source is crucial to prevent crashes or instability.",
            "Raspberry Pi can be used for home automation with platforms like Home Assistant or OpenHAB. You can use "
            "the GPIO pins to control lights, fans, and other devices. You'll need to install the appropriate "
            "software and configure the devices.",
            "To connect Raspberry Pi to Wi-Fi, click on the Wi-Fi icon in the desktop environment and select your "
            "network. Alternatively, you can configure Wi-Fi by editing the `wpa_supplicant.conf` file on the SD card "
            "before booting the Pi.",
            "If your Raspberry Pi is running slowly, consider checking the CPU usage and temperature using tools like "
            "`htop` or `vcgencmd`. If the issue persists, try optimizing your code, reducing unnecessary processes, "
            "or checking for power supply issues."
        ]

        d_reps = l2v.encode(documents)

        # Compute cosine similarity
        q_reps_norm = torch.nn.functional.normalize(q_reps, p=2, dim=1)
        d_reps_norm = torch.nn.functional.normalize(d_reps, p=2, dim=1)
        cos_sim = torch.mm(q_reps_norm, d_reps_norm.transpose(0, 1))

        # Loop through each row of the cosine similarity matrix
        correct_count = 0
        for i in range(cos_sim.shape[0]):
            # Find the index of the maximum similarity value for each query
            max_sim_index = torch.argmax(cos_sim[i]).item()

            # Compare with the intended index (the one that should be correct for the query)
            if max_sim_index == intended_indices[i]:
                correct_count += 1

            # Display the query, the predicted document index, and whether it matches the intended index
            print(f"Query: {queries[i]}")
            print(f"Predicted document index: {max_sim_index}, Intended index: {intended_indices[i]}")
            print(f"Match: {'Yes' if max_sim_index == intended_indices[i] else 'No'}")
            print("-" * 50)

        # Print the overall result
        print(f"\nTotal matches: {correct_count}/{len(queries)}")
        print(torch.argmax(cos_sim, axis=1))
        print(cos_sim)


if __name__ == '__main__':
    unittest.main()

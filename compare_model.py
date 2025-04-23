import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from llm2vec import LLM2Vec

import LLM2VecModel


def main():
    print("Cuda is ", torch.cuda.is_available())
    MODEL_PATH_OR_NAME_A = r"meta-llama/Llama-3.2-1B"
    MODEL_PATH_OR_NAME_B = r"./saved_models"
    BEST_MODEL_BASELINE = "McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp"

    model_a = LLM2Vec.from_pretrained(MODEL_PATH_OR_NAME_A)
    model_b = LLM2Vec.from_pretrained(MODEL_PATH_OR_NAME_B)

    model_best = LLM2VecModel.LLM2VecModel(BEST_MODEL_BASELINE)

    print("manuals_test")
    result_best, result_a, result_b = run_test(manuals_test, [model_best, model_a, model_b])
    evaluate_topk_accuracy([result_a, result_b, result_best])

    print("manuals_seen_test")
    result_best, result_a, result_b = run_test(manuals_seen_test, [model_best, model_a, model_b])
    evaluate_topk_accuracy(result_a, result_b)

    print("manuals_unseen_test")
    result_best, result_a, result_b = run_test(manuals_unseen_test, [model_best, model_a, model_b])
    compare_similarity(result_a, result_b)

    print("Similarity Move to the best")
    print(torch.argmax(result_best, dim=1))
    vis_similiarty(abs(result_best - result_b) - abs(result_best - result_a))
    evaluate_topk_accuracy([result_a, result_b, result_best])
    plt.show()


def run_test(test, models):
    output = []
    for model in models:
        output.append(test(model))
    return output


def evaluate_topk_accuracy(similarity_matrices, topk=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)):
    results = []

    for idx, sim_matrix in enumerate(similarity_matrices):
        num_queries = sim_matrix.size(0)
        topk_preds = sim_matrix.topk(max(topk), dim=1).indices

        matrix_result = {
            "matrix_index": idx + 1,
            "total": num_queries,
        }

        print(f"Matrix {idx + 1}:")

        for k in topk:
            correct_topk = torch.stack([
                (topk_preds[i, :k] == i).any() for i in range(num_queries)
            ]).sum().item()

            acc = correct_topk / num_queries
            matrix_result[f"top{k}_accuracy"] = acc
            matrix_result[f"correct_top{k}"] = correct_topk

            print(f"  Top-{k} Accuracy: {acc:.2%} ({correct_topk} / {num_queries})")

        print()
        results.append(matrix_result)

    return results


def calculate_cos_similarity(embeder, docs, queries):
    q_reps = embeder.encode(queries)
    d_reps = embeder.encode(docs)

    q_reps_norm = torch.nn.functional.normalize(q_reps, p=2, dim=1)
    d_reps_norm = torch.nn.functional.normalize(d_reps, p=2, dim=1)
    return torch.mm(q_reps_norm, d_reps_norm.transpose(0, 1))


def titans_test(a):
    prompts = [
        "How do creatures on Titan communicate?",
        "What is the lost city of Auravelle?",
        "Describe the properties of Nocturnium.",
        "How do deep-sea creatures communicate?",
        "Are there ancient cities under ice?",
        "Do any minerals glow naturally?",
        "What is Titan known for?",
        "What are the legends of Atlantis?",
        "How do solar panels absorb energy?",
        "What is the capital of France?",
        "How do birds migrate?",
        "What is quantum entanglement?"
    ]

    documents = [
        "Scientists believe that bioluminescent organisms in Titan’s methane seas use rhythmic flashes of light as a "
        "form of communication.",
        "The lost city of Auravelle is a rumored civilization buried beneath the Arctic ice, said to contain ancient "
        "structures resistant to extreme cold.",
        "Nocturnium is a rare mineral that absorbs moonlight and emits a faint glow only in complete darkness.",
        "Many deep-sea creatures use bioluminescence to attract prey or signal to others in their environment.",
        "Some theories suggest that lost civilizations could be buried beneath glaciers, but no definitive proof has "
        "been found.",
        "Some minerals, like fluorite and phosphorescent rocks, can emit light after being exposed to energy sources.",
        "Titan is Saturn’s largest moon, known for its thick atmosphere and methane lakes.",
        "Atlantis is a mythical island city described by Plato, said to have sunk into the ocean due to divine "
        "punishment.",
        "Solar panels use photovoltaic cells to convert sunlight into electricity, storing it for later use.",
        "Paris is the capital of France, known for its rich history and cultural landmarks.",
        "Many bird species migrate seasonally, using the Earth’s magnetic field and landmarks for navigation.",
        "Quantum entanglement is a phenomenon where two particles remain interconnected, instantly affecting each "
        "other regardless of distance."
    ]

    return calculate_cos_similarity(embeder=a, docs=documents, queries=prompts)


def manuals_seen_test(a):
    prompts = [
        "What type of electrical connection is required for the Over-the-Range Microwave models EI30BM55H?",
        "What should you do before cleaning the oven to ensure safety?",
        "What should you do before cleaning the fridge-freezer to ensure safety?",
        "What should you do if installing the RH36WC55GS hood in a cold weather location?",
        "What should you do before cleaning or carrying out maintenance on the Electrolux Premier Cooker Hood?",
        "What should you always do before removing the protective cover from the Elinchrom BX flash unit?",
        "What feature helps reduce sound in the ILCGR4822 series sinks?",
        "What should you do before connecting the power to the Touchmonitor?",
        "What types of media can the Model 2382IP Design Music Center play?",
        "What are the primary maintenance steps for the Model 499A OZ sensor?"
    ]

    documents = [
        "They require a 120V, 60Hz grounded electrical outlet with a dedicated 15- or 20-amp circuit.",
        "Before cleaning the oven, you should switch it off and allow it to cool down completely to avoid the risk of "
        "burns or electric shock.",
        "Before cleaning the fridge-freezer, you should switch it off and unplug it from the power supply to avoid "
        "the risk of electric shock.",
        "For cold weather installations, you should follow the instructions for installing a thermal break to prevent "
        "cold air from entering through the duct.",
        "You should isolate the appliance from the electricity supply before cleaning or performing any maintenance.",
        "You should switch the unit OFF and disconnect it from the fully earthed (grounded) power outlet before "
        "removing the protective cover.",
        "The ILCGR4822 sinks have a heavy sound-deadening coating that helps reduce sound.",
        "Before connecting the power, you should ensure that both the PC and the Touchmonitor are turned off.",
        "The Model 2382IP can play CDs, USB drives, SD cards, and MP3 files. It also includes an iPod docking station "
        "for playback.",
        "The main maintenance tasks include cleaning the membrane every four to six months and replacing the "
        "electrolyte solution and membrane if performance degrades. This involves unscrewing the sensor, inspecting "
        "and polishing the cathode if needed, and carefully refilling the electrolyte while avoiding air bubbles."
    ]

    return calculate_cos_similarity(embeder=a, docs=documents, queries=prompts)


def manuals_unseen_test(a):
    prompts = [
        "What batteries does the king Arthur chess set use?",
        "What happens when the LASER button is pressed and then released on the remote?",
        "What is the purpose of the Twin Iris Function on the SANYO PLV-Z4?",
        "What happens when you press the top of the car on the Fisher-Price H4088 toy?",
        "How long should you charge the battery the first time you use the Power Wheels vehicle?",
        "How long is the Fluke Remote covered under its limited warranty?",
        "What are two ways you can connect to the FortiMail web-based interface?",
        "What is the role of the FortiBridge when the FortiGate is operating normally?",
        "What type of security services does the FortiGate-100A provide?",
        "What should you do before cleaning the wine cooler to avoid electric shock?",
        "Why should you read the manual before using the product?",
        "Why should users thoroughly read the information provided in the documentation?",
        "What happens when you press the [CH16] button on the FM-8700 VHF Radiotelephone?",
        "What temperature range does the Heavy-Duty FDO thermostat on the H280 operate within?",
        "What advantage does the FDO thermostat have over the lower-priced BJWA thermostat?"
    ]
    documents = [
        "Arthur Chess uses AA batteries and or rechargeable batteries.",
        "With the LASER pressed, the light turns on; when the LASER is released, the light turns off.",
        "The Twin Iris Function adjusts the brightness and contrast of the image.",
        "Pressing the top of the car activates lights and sound effects.",
        "You should charge the battery for at least 18 hours before using the vehicle for the first time.",
        "It is covered for three years from the date of purchase.",
        "You can connect to the FortiMail web-based interface using a cross-over Ethernet cable directly to a "
        "computer or a straight-through Ethernet cable through a hub or switch.",
        "When the FortiGate is operating normally, the FortiBridge passes traffic through the FortiGate to apply "
        "security services such as firewall, IPS, and antivirus scanning.",
        "The FortiGate-100A provides firewall, VPN, antivirus, intrusion prevention (IPS), antispam, "
        "and web filtering services to protect networks from various threats.",
        "Before cleaning, you should turn the temperature control to OFF and unplug the cooler from the power supply "
        "to avoid electric shock.",
        "Because if the product is not used correctly, it may cause unexpected injury to users or bystanders.",
        "To ensure proper understanding and operation of the system, prevent harm to users or damage to property, "
        "and keep the information available for future reference.",
        "Pressing the [CH16] button sets the FM-8700 to Channel 16, the international calling and distress channel.",
        "The Heavy-Duty FDO thermostat on the H280 operates within a temperature range of 150°F to 500°F (66°C to "
        "260°C).",
        "The FDO thermostat offers better temperature-holding performance and avoids the temperature creep of up to "
        "90°F seen in the lower-priced BJWA thermostat."
    ]

    return calculate_cos_similarity(embeder=a, docs=documents, queries=prompts)


def manuals_test(a):
    prompts = [
        "How do I reset my router to factory settings?",
        "What is the proper way to clean a coffee machine?",
        "How do I install a new printer driver on Windows?",
        "What should I do if my smartphone won't charge?",
        "How can I update the firmware on my smart TV?",
        "How do I calibrate my home thermostat?",
        "What’s the correct way to defrost a freezer?",
        "How do I connect Bluetooth headphones to a laptop?",
        "How can I replace the battery in a smoke detector?",
        "How do I set up parental controls on a tablet?",
        "What should I do if my washing machine won’t drain?",
        "How do I perform a factory reset on my smartwatch?",
        "What is the safest way to clean a flat-screen TV?",
        "How do I troubleshoot a Wi-Fi connection issue?",
        "What steps are needed to back up data on an Android phone?",
        "How do I install a new graphics card in my PC?",
        "How do I clean my robotic vacuum sensors?",
        "How can I set up voice control for my smart home?",
        "How do I replace the ink cartridge in my printer?",
        "What’s the best way to extend my laptop battery life?",
        "How do I set a schedule on a smart thermostat?",
        "How do I connect a wireless keyboard to my computer?",
        "What should I do if my dishwasher won't start?",
        "How can I enable dark mode on my phone?",
        "How do I upgrade the RAM on my desktop PC?",
        "How do I configure email on a tablet?",
        "What’s the best way to store passwords securely?",
        "How do I clean and maintain a mechanical keyboard?",
        "How do I connect a second monitor to my laptop?",
        "What should I do if my screen is flickering?",
        "How do I replace a broken phone screen?",
        "How can I tell if my smoke detector is working?",
        "What should I do if my laptop overheats?",
        "How do I turn off push notifications on Android?",
        "How do I back up files to an external hard drive?",
        "How do I install antivirus software?",
        "How can I improve my Wi-Fi signal at home?",
        "How do I use a surge protector correctly?",
        "How can I organize cables behind my desk?",
        "What’s the best way to clean my headphones?",
        "How do I test a remote control battery?",
        "How do I connect a projector to a laptop?",
        "How can I activate airplane mode?",
        "How do I reset my email password?",
        "How do I change the language settings on my device?",
        "What should I do if an app keeps crashing?",
        "How can I factory reset a smart speaker?",
        "How do I know if my phone is waterproof?",
        "How do I clean a tablet screen?",
        "What’s the safest way to remove a USB drive?"
    ]

    documents = [
        "To reset your router, hold the reset button for 10 seconds until the lights blink.",
        "Clean your coffee machine with a 1:1 solution of vinegar and water, followed by a rinse cycle.",
        "Download the latest driver from the printer manufacturer’s site and follow installation steps.",
        "Try a new charger, inspect the port for debris, and reboot the phone. Consider a factory reset if needed.",
        "Navigate to Settings > About > Software Update on your smart TV to install firmware updates.",
        "Go into the thermostat settings, choose 'calibrate', and follow on-screen instructions.",
        "Unplug the freezer and allow it to sit with the door open until the ice melts.",
        "Enable Bluetooth on your laptop and headphones, then pair them from the available devices list.",
        "Detach the smoke detector, open the battery compartment, and insert a new battery with correct polarity.",
        "Go to tablet Settings > Digital Wellbeing or Parental Controls and configure restrictions.",
        "Check for clogs in the drain hose or pump filter, then run the drain cycle again.",
        "Go to your smartwatch settings > System > Reset, and follow the confirmation steps.",
        "Use a dry microfiber cloth or one lightly dampened with water to clean the screen gently.",
        "Restart your router, reconnect your device, and update your network drivers if necessary.",
        "Open Settings > System > Backup on your Android device and activate cloud backup.",
        "Shut down your PC, open the case, remove the old graphics card, and install the new one into the PCIe slot.",
        "Wipe the robotic vacuum sensors with a soft dry cloth. Avoid using liquids directly.",
        "Activate voice assistant features in your app, then assign devices to rooms for voice control.",
        "Lift the printer cover, remove the old cartridge, and insert a new one until it clicks.",
        "Lower brightness, disable background apps, and use battery saver mode to extend battery life.",
        "Access the thermostat app, create time slots, and assign temperature settings for each slot.",
        "Turn on Bluetooth, insert the USB dongle if needed, and press the keyboard’s pairing button.",
        "Ensure the dishwasher door is shut, check for power, and inspect the control panel lights.",
        "Go to Display settings and choose 'Dark Mode' or 'Dark Theme' under Appearance.",
        "Power off your PC, locate the RAM slots, and carefully insert new modules until they click.",
        "Open the Mail app, enter your email and password, and follow server setup prompts.",
        "Use a password manager app or enable built-in browser password storage with two-factor authentication.",
        "Remove keycaps, use compressed air to clean under them, and wipe the case with alcohol wipes.",
        "Plug in the second monitor via HDMI or DisplayPort, then use display settings to extend your screen.",
        "Try updating graphics drivers and changing refresh rate settings to stop screen flickering.",
        "Use a screen replacement kit or take the phone to a certified technician for a safe repair.",
        "Press the test button on the smoke detector; it should emit a loud beep if working.",
        "Make sure vents are clear, reduce heavy processes, and consider a cooling pad for overheating laptops.",
        "Open Settings > Notifications and toggle off push alerts for individual apps.",
        "Connect the drive to your computer, then drag and drop files or use backup software.",
        "Download antivirus software, run the installer, and follow setup instructions to enable real-time protection.",
        "Relocate your router, use a Wi-Fi extender, and reduce interference from walls or other electronics.",
        "Plug sensitive devices into the surge protector, ensure it's turned on, and test its indicator light.",
        "Use cable sleeves or zip ties to bundle cords together and label each one for easy access.",
        "Remove debris with a dry cloth, use cotton swabs for crevices, and avoid water contact.",
        "Point the remote at a camera (like on a phone), press buttons, and check for IR light via the screen.",
        "Connect the projector via HDMI or VGA, select the correct input source, and mirror your display.",
        "Swipe down and tap the airplane icon, or toggle it from the Settings > Network menu.",
        "Go to your email provider’s website, click 'Forgot Password', and follow the recovery instructions.",
        "Go to Settings > Language & Input, select your desired language, and restart if prompted.",
        "Clear the app cache/data or reinstall the app if it continues crashing frequently.",
        "Hold the mute and volume-down buttons until the light blinks, indicating a factory reset.",
        "Check the manufacturer’s specifications or SIM tray seal for waterproof ratings like IP68.",
        "Use a microfiber cloth and screen-safe cleaner to gently remove smudges from your tablet.",
        "Eject the USB device via system tray/safely remove option before unplugging it from the port."
    ]

    return calculate_cos_similarity(embeder=a, docs=documents, queries=prompts)


def compare_similarity(similarity_a, similarity_b):
    print("Similarity A")
    print(similarity_a)
    print("Similarity B")
    print(similarity_b)
    print("Predictions A ")
    print(torch.argmax(similarity_a, dim=1))
    print("Predictions B")
    print(torch.argmax(similarity_b, dim=1))
    plot_matrix_heatmap(similarity_a)
    plot_matrix_heatmap(similarity_b)

    print("Different of similarity B - A")
    print(similarity_b - similarity_a)
    plot_matrix_heatmap(similarity_b - similarity_a)


def vis_similiarty(similarity):
    print(similarity)
    print(torch.argmax(similarity, dim=1))
    plot_matrix_heatmap(similarity)


def cloud_test(model):
    documents = [
        "Clouds are formed when water vapor in the atmosphere cools and condenses into visible water droplets or ice "
        "crystals.",
        "There are different types of clouds, such as cumulus, stratus, and cirrus, which vary in appearance and "
        "altitude.",
        "Cloud computing refers to the delivery of computing services like storage, databases, and software over the "
        "internet.",
        "Clouds can have a significant impact on weather patterns and climate conditions."
    ]

    prompts = [
        "What causes clouds to form?",
        "Explain the different types of clouds.",
        "What is cloud computing?",
        "How do clouds affect the weather?"
    ]

    return calculate_cos_similarity(embeder=model, docs=documents, queries=prompts)


def plot_matrix_heatmap(matrix):
    """
    Plots a heatmap of a given matrix where negative values are red
    and positive values are blue, with intensity based on magnitude.

    Parameters:
    - matrix: A 2D NumPy array or list of lists

    """

    if isinstance(matrix, torch.Tensor):
        matrix = matrix.cpu().detach().numpy()  # Convert to NumPy safely
    # Convert to numpy array in case it's not
    matrix = np.array(matrix)

    # Define a diverging colormap (blue for positive, red for negative)
    cmap = sns.diverging_palette(240, 10, as_cmap=True)

    # Create the heatmap
    plt.figure(figsize=(6, 6))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap=cmap, center=0, linewidths=0.5, cbar=True)

    # Set title and show the plot
    plt.title("Matrix Heatmap")


if __name__ == '__main__':
    main()

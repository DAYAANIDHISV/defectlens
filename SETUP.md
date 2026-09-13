# Setting up DefectLens — step by step, for someone who has never done this before

There are two ways. **Way A** needs nothing installed and takes about ten minutes. **Way B** puts the
inspection station (the web page you demo) on your own computer; it takes about twenty minutes the
first time and one minute every time after.

---

## Way A · Run it in Google Colab (nothing to install)

Colab is a free Google website that runs code for you on Google's computers.

1. Open this link: **https://colab.research.google.com/github/DAYAANIDHISV/defectlens/blob/main/DefectLens_colab.ipynb**
2. Sign in with any Google account if it asks.
3. At the top, click **Runtime → Change runtime type**, choose **T4 GPU**, click **Save**.
4. Click **Runtime → Run all**. If a box says the notebook was not written by Google, click **Run anyway**.
5. Wait about 6–8 minutes and scroll down. You will see, in order: the photos being disturbed, the
   model training, the stress test table (naive model vs ours), the CSV, and the heat-maps.

That's it. Nothing is saved on your computer.

---

## Way B · The inspection station on your own computer

You will do four things once: **install Python**, **download the project**, **install the
add-ons it needs**, **download our trained models**. Then you start it with one command.

Pick your computer: [Mac](#on-a-mac) or [Windows](#on-windows).

### On a Mac

**1. Install Python** (the programming language the project is written in)

1. Go to **https://www.python.org/downloads/** and click the yellow **Download Python** button.
2. Open the downloaded file from your Downloads folder and click **Continue → Agree → Install**.
3. Check it worked: press **⌘ Space**, type **Terminal**, press **Return**. A text window opens —
   this is the Terminal, where you type commands. Type this and press **Return**:
   ```
   python3 --version
   ```
   You should see something like `Python 3.12.6`. Any number 3.10 or higher is fine.

**2. Download the project**

1. Open **https://github.com/DAYAANIDHISV/defectlens**
2. Click the green **Code** button → **Download ZIP**.
3. Open your **Downloads** folder and double-click **defectlens-main.zip**. You now have a folder called
   **defectlens-main**.

**3. Go into the project folder in the Terminal**

In the Terminal, type `cd ` (the letters c, d and a space — don't press Return yet), then **drag the
defectlens-main folder from Finder into the Terminal window**, and press **Return**. Dragging types
the folder's location for you.

**4. Install the add-ons** (one time only — a few minutes, it downloads a few hundred MB)

Copy these two lines into the Terminal, one at a time, pressing **Return** after each:
```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```
The first makes a private box (`.venv`) for this project's add-ons, so they cannot clash with anything
else on your computer. The second fills it. Wait until the Terminal shows the prompt again.

**5. Download our three trained models** (one time only, about 135 MB)

Copy this into the Terminal and press **Return**:
```
for m in final final-s1 final-s2; do curl -L -o models/$m.pt https://github.com/DAYAANIDHISV/defectlens/releases/download/v1.0/$m.pt; done
```
*Or by clicking:* open **https://github.com/DAYAANIDHISV/defectlens/releases/tag/v1.0**, click
`final.pt`, `final-s1.pt` and `final-s2.pt` under **Assets** to download them, then drag all three into
the **models** folder inside **defectlens-main**.

**6. Start the station**

```
.venv/bin/python app.py
```
Wait until you see `open http://localhost:8050` (a red "development server" warning is normal — ignore
it). Now open your web browser and go to **http://localhost:8050**.

**Leave the Terminal window open** while you use the station. To stop it, click the Terminal and press
**Control C**.

**Next time:** open the Terminal, do step 3 again (`cd` + drag the folder), then step 6. Steps 1, 2, 4
and 5 are never repeated.

### On Windows

**1. Install Python**

1. Go to **https://www.python.org/downloads/** and click **Download Python**.
2. Run the downloaded file. **On the first screen, tick "Add python.exe to PATH"** at the bottom — this
   matters — then click **Install Now**.
3. Check it worked: click **Start**, type **PowerShell**, press **Enter**. In the blue window, type
   this and press **Enter**:
   ```
   py --version
   ```
   You should see something like `Python 3.12.6`. Any number 3.10 or higher is fine.

**2. Download the project**

1. Open **https://github.com/DAYAANIDHISV/defectlens**
2. Click the green **Code** button → **Download ZIP**.
3. In your **Downloads** folder, right-click **defectlens-main.zip** → **Extract All** → **Extract**.
4. Open the new folder until you can see files called **README.md** and **app.py**. That is the
   project folder. (Windows often puts it one folder deeper: `defectlens-main\defectlens-main`.)

**3. Go into the project folder in PowerShell**

Easiest: in File Explorer, open the project folder, click the address bar at the top, type
**powershell**, press **Enter**. A PowerShell window opens already inside the folder.

**4. Install the add-ons** (one time only — a few minutes)

```
py -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

**5. Download our three trained models** (one time only, about 135 MB)

```
foreach ($m in "final","final-s1","final-s2") { curl.exe -L -o "models\$m.pt" "https://github.com/DAYAANIDHISV/defectlens/releases/download/v1.0/$m.pt" }
```
*Or by clicking:* open **https://github.com/DAYAANIDHISV/defectlens/releases/tag/v1.0**, download the
three `.pt` files under **Assets**, and move them into the project's **models** folder.

**6. Start the station**

```
.venv\Scripts\python app.py
```
If Windows asks whether Python may use the network, click **Allow**. Then open your browser at
**http://localhost:8050**. Keep the PowerShell window open; press **Control C** in it to stop.

**Next time:** open PowerShell in the project folder (step 3), then step 6.

---

## Using the station

- **Drag a photo** onto the grey box, or click it to choose a file. You get **PASS**, **REJECT** (and
  which defect) or **REVIEW** (not sure — a person should look), the chance for each class, and a
  heat-map of where the model looked.
- **Stress it** buttons change the photo — turn it, dim the light, change the background, add glare —
  and inspect it again. They stack; **Back to the original** undoes them.
- **Inspect a whole folder…** runs every photo in a folder and lets you download `submission.csv`.

**Need photos to try?** Download the organisers' practice package:
**https://github.com/sanjai-umashankar/AI-Arena-AIML-Hackathon-2026/raw/main/AI_ARENA_PARTICIPANT.zip**,
unzip it, and use any image from `PARTICIPANT_PACKAGE/DefectLens/validation/`. If you copy that whole
**DefectLens** folder into the project's **data** folder (create it), the station's sample buttons appear
too.

---

## If something goes wrong

| What you see | What it means, and what to do |
|---|---|
| `command not found: python3` (Mac) or `py is not recognized` (Windows) | Python is not installed, or on Windows the "Add python.exe to PATH" box was not ticked. Install again (step 1). |
| `No such file or directory: requirements.txt` | The Terminal is not inside the project folder. Do step 3 again. |
| `No trained models yet` | The three model files are not in the **models** folder. Do step 5. |
| `Address already in use` | The station is already running in another window. Use that one, or close it first. |
| The browser says it can't connect | The station is not running — start it (step 6) and keep its window open. |
| `ERROR: Could not find a version that satisfies the requirement torch==2.9.0` | Your Python is too old or too new for that version. Install Python 3.12 from python.org (step 1) and start again from step 4. |
| The same error on an **older Intel Mac** (bought before late 2020) | PyTorch no longer makes versions for Intel Macs. Use Way A (Colab) on that computer. |

The Windows steps are written carefully but were tested only on a Mac; if a step fails on Windows,
Way A (Colab) always works.

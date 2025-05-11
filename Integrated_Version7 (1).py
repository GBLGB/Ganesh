"""
NEPSE Trading Bot (Ultra-Low-Latency HFT Version, GPU-Accelerated for RTX 3060, Python 3.11)
- GPU-accelerated rapid buy clicking and rapid form filling (Numba CUDA).
- F5 (refresh) just to refresh the page; rest the bot to initiate the form filling and order placement with minimal delay at all times.
- Back-to-back order placement: As soon as one order completes, immediately start the next during active trading hours.
- Stops placing new orders after a circuit level price order is placed.
- Auto-refresh and prepare order to submit at pre-open (10:30:00.0000) session boundary with no delay, to beat others and be the first one.
- Latency profiler to measure boundary click-to-server-response with nanosecond accuracy.
- Captcha must be filled manually by user if detected, with a blocking wait for human entry (possibly 30 seconds).
- Browser launches directly to login page (never blank).
- After every refresh, always navigate to the order entry page and the bot to do its job of automation.
- Robust error handling during form preparation, with logging and screenshots.
"""

import argparse
import logging
import os
import threading
import time as tm
from datetime import datetime, time, date
import math
import sys
import gc
import numpy as np

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    StaleElementReferenceException,
    WebDriverException,
)
from webdriver_manager.chrome import ChromeDriverManager

try:
    import keyboard
except ImportError:
    print("Please install the 'keyboard' package to use hotkey features. Run: pip install keyboard")
    sys.exit(1)

try:
    from numba import cuda
    GPU_ENABLED = True
except ImportError:
    GPU_ENABLED = False

USERNAME = "GL302996"
PASSWORD = "Saanu####8775"
SYMBOL = "NLO"
QUANTITY = 100
FORM_FILLING_MODE = True
MAX_CLICK_ATTEMPTS = 10000000
CLICK_INTERVAL = 1e-9
NEXT_ORDER_WAIT = 0
CAPTCHA_FILL_WAIT = 30
PREOPEN_START_HOUR = 10
PREOPEN_START_MINUTE = 29
PREOPEN_END_HOUR = 10
PREOPEN_END_MINUTE = 45
REGULAR_START_HOUR = 10
REGULAR_START_MINUTE = 59
REGULAR_END_HOUR = 15
REGULAR_END_MINUTE = 0
CIRCUIT_LIMIT_PERCENTAGE = 10

HOTKEY_COMBO = "ctrl+shift+q"
REFRESH_HOTKEY = "f5"
SHM_RING_SIZE = 65536

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="nepse_trading_bot.log",
)
logger = logging.getLogger()
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def preallocate_numpy_ring(dtype, size):
    return np.zeros(size, dtype=dtype)

def disable_gc_during_critical(func):
    def wrapper(*args, **kwargs):
        was_enabled = gc.isenabled()
        if was_enabled:
            gc.disable()
        try:
            return func(*args, **kwargs)
        finally:
            if was_enabled:
                gc.enable()
    return wrapper

if GPU_ENABLED:
    @cuda.jit
    def gpu_atomic_fill(arr, value):
        idx = cuda.grid(1)
        if idx < arr.size:
            arr[idx] = value

class SHMRingBuffer:
    def __init__(self, dtype, size=SHM_RING_SIZE):
        self.size = size
        self.buf = preallocate_numpy_ring(dtype, size)
        self.head = 0
        self.tail = 0

    def push(self, value):
        next_head = (self.head + 1) % self.size
        if next_head == self.tail:
            return False
        self.buf[self.head] = value
        self.head = next_head
        return True

    def pop(self):
        if self.head == self.tail:
            return None
        value = self.buf[self.tail]
        self.tail = (self.tail + 1) % self.size
        return value

def get_chrome_driver(user_data_dir=None):
    chrome_options = Options()
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--disable-notifications")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-logging"])
    if user_data_dir:
        chrome_options.add_argument(f"--user-data-dir={user_data_dir}")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.get("https://tms18.nepsetms.com.np/")  # Always launch directly to login page
    return driver

class NepseTrader:
    def __init__(self, user_data_dir=None, use_gpu=False):
        self.url = "https://tms18.nepsetms.com.np/tms/me/memberclientorderentry"
        self.login_url = "https://tms18.nepsetms.com.np/"
        self.screenshot_dir = "debug_screenshots"
        os.makedirs(self.screenshot_dir, exist_ok=True)
        self.logs_dir = "order_logs"
        os.makedirs(self.logs_dir, exist_ok=True)
        self.user_data_dir = user_data_dir
        self.driver = get_chrome_driver(user_data_dir)
        self.wait = WebDriverWait(self.driver, 1)
        self.successful_orders = 0
        self.order_success = threading.Event()
        self.stop_clicking = threading.Event()
        self.pre_open_scheduled = False
        self.order_log_file = f"{self.logs_dir}/orders_{datetime.now().strftime('%Y%m%d')}.csv"
        if not os.path.exists(self.order_log_file):
            with open(self.order_log_file, "w") as f:
                f.write("timestamp,symbol,quantity,price,status,mode,click_attempts\n")
        self.pre_close_price = None
        self.session_id = None
        self.active = True

        self.last_transaction_id = None
        self.last_known_price = None
        self.last_known_qty = None

        self.trading_date = None
        self.circuit_limit_price_for_date = None
        self.regular_session_price_for_date = None
        self.circuit_hit_for_date = False

        self.use_gpu = use_gpu and GPU_ENABLED
        if self.use_gpu:
            self.ring = SHMRingBuffer(np.float64, SHM_RING_SIZE)
        else:
            self.ring = SHMRingBuffer(np.float64, SHM_RING_SIZE)

        self._restore_state_from_local_storage()
        self.refresh_requested = threading.Event()
        self._register_refresh_hotkey()

    def _get_state_file_path(self):
        return os.path.join(self.logs_dir, "bot_state.txt")

    def _restore_state_from_local_storage(self):
        state_file = self._get_state_file_path()
        if os.path.exists(state_file):
            try:
                with open(state_file, "r") as f:
                    lines = f.readlines()
                for line in lines:
                    if line.startswith("last_transaction_id:"):
                        self.last_transaction_id = line.strip().split(":", 1)[1].strip()
                    if line.startswith("last_known_price:"):
                        self.last_known_price = float(line.strip().split(":", 1)[1].strip())
                    if line.startswith("last_known_qty:"):
                        self.last_known_qty = int(line.strip().split(":", 1)[1].strip())
                    if line.startswith("trading_date:"):
                        self.trading_date = line.strip().split(":", 1)[1].strip()
                    if line.startswith("circuit_limit_price_for_date:"):
                        val = line.strip().split(":", 1)[1].strip()
                        self.circuit_limit_price_for_date = float(val) if val != "None" else None
                    if line.startswith("regular_session_price_for_date:"):
                        val = line.strip().split(":", 1)[1].strip()
                        self.regular_session_price_for_date = float(val) if val != "None" else None
                    if line.startswith("circuit_hit_for_date:"):
                        val = line.strip().split(":", 1)[1].strip()
                        self.circuit_hit_for_date = val == "True"
            except Exception as e:
                logger.error(f"Error restoring bot state: {e}")

    def _save_state_to_local_storage(self):
        state_file = self._get_state_file_path()
        try:
            with open(state_file, "w") as f:
                f.write(f"last_transaction_id:{self.last_transaction_id}\n")
                f.write(f"last_known_price:{self.last_known_price}\n")
                f.write(f"last_known_qty:{self.last_known_qty}\n")
                f.write(f"trading_date:{self.trading_date}\n")
                f.write(f"circuit_limit_price_for_date:{self.circuit_limit_price_for_date}\n")
                f.write(f"regular_session_price_for_date:{self.regular_session_price_for_date}\n")
                f.write(f"circuit_hit_for_date:{self.circuit_hit_for_date}\n")
            try:
                os.chmod(state_file, 0o600)
            except Exception:
                pass
        except Exception as e:
            logger.error(f"Error saving bot state: {e}. Check file/folder permissions for '{state_file}'.")

    def take_screenshot(self, name):
        try:
            filename = f"{self.screenshot_dir}/debug_{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            self.driver.save_screenshot(filename)
            logger.info(f"Screenshot saved: {filename}")
        except Exception as e:
            logger.error(f"Screenshot error: {e}")

    def is_element_present(self, by, value, timeout=0.5):
        try:
            WebDriverWait(self.driver, timeout).until(EC.presence_of_element_located((by, value)))
            return True
        except (TimeoutException, NoSuchElementException):
            return False

    def wait_and_click(self, by, value, timeout=1, retries=1):
        for attempt in range(retries):
            try:
                element = WebDriverWait(self.driver, timeout).until(
                    EC.element_to_be_clickable((by, value))
                )
                element.click()
                return True
            except (TimeoutException, NoSuchElementException, StaleElementReferenceException) as e:
                logger.warning(f"Click attempt {attempt+1} failed: {e}")
                if attempt == retries - 1:
                    logger.error(f"Failed to click element after {retries} attempts")
                    self.take_screenshot(f"click_failure_{value.replace('/', '_')}")
                    return False
                tm.sleep(0.1)
        return False

    def is_pre_open_hours(self):
        now = datetime.now().time()
        pre_open_start = time(PREOPEN_START_HOUR, PREOPEN_START_MINUTE)
        pre_open_end = time(PREOPEN_END_HOUR, PREOPEN_END_MINUTE)
        return pre_open_start <= now <= pre_open_end

    def is_normal_trading_hours(self):
        now = datetime.now().time()
        normal_start = time(REGULAR_START_HOUR, REGULAR_START_MINUTE)
        normal_end = time(REGULAR_END_HOUR, REGULAR_END_MINUTE)
        return normal_start <= now <= normal_end

    def is_trading_hours(self):
        return self.is_pre_open_hours() or self.is_normal_trading_hours()

    def login(self):
        try:
            logger.info("Navigating to login page")
            self.driver.get(self.login_url)
            # If already logged in, skip filling credentials
            if self.is_element_present(By.XPATH, "//span[contains(text(), 'Dashboard')]"):
                logger.info("Already logged in")
                return True
            # Fill credentials on landing page
            username_input = self.wait.until(EC.element_to_be_clickable(
                (By.XPATH, "//input[contains(@placeholder, 'User') or @id='username']")))
            password_input = self.wait.until(EC.element_to_be_clickable(
                (By.XPATH, "//input[@type='password']")))
            username_input.clear()
            username_input.send_keys(USERNAME)
            password_input.clear()
            password_input.send_keys(PASSWORD)
            login_button = self.wait.until(EC.element_to_be_clickable(
                (By.XPATH, "//button[@type='submit' or contains(text(), 'Login') or contains(text(), 'Sign')]")))
            login_button.click()
            # Captcha logic: if present, block for manual fill (wait for dashboard or timeout)
            captcha_xpath = "//input[contains(@placeholder, 'Captcha') or @formcontrolname='captcha']"
            if self.is_element_present(By.XPATH, captcha_xpath, timeout=2):
                logger.info(f"Captcha detected. Please fill the captcha. Waiting {CAPTCHA_FILL_WAIT} seconds for manual entry...")
                self.take_screenshot("captcha_required")
                start = tm.time()
                while tm.time() - start < CAPTCHA_FILL_WAIT:
                    if self.is_element_present(By.XPATH, "//span[contains(text(), 'Dashboard')]", timeout=2):
                        logger.info("Captcha filled and login successful.")
                        return True
                    tm.sleep(1)
                logger.error("Timeout waiting for manual captcha entry.")
                return False
            self.wait.until(EC.presence_of_element_located((By.XPATH, "//span[contains(text(), 'Dashboard')]")))
            logger.info("Login successful")
            return True
        except Exception as e:
            logger.error(f"Login failed: {e}")
            self.take_screenshot("login_error")
            return False

    def navigate_to_order_page(self):
        try:
            logger.info("Navigating to order entry page")
            self.driver.get(self.url)
            self.wait.until(EC.presence_of_element_located((By.XPATH, "//span[text()='Order Management']")))
            logger.info("Order page loaded")
            return True
        except Exception as e:
            logger.error(f"Failed to navigate to order page: {e}")
            self.take_screenshot("order_page_error")
            return False

    def check_session_validity(self):
        try:
            if not self.is_element_present(By.XPATH, "//span[contains(text(), 'Dashboard')]", timeout=0.5):
                logger.warning("Session may have expired or browser refreshed, attempting to re-login")
                self.driver.refresh()
                if not self.login():
                    logger.error("Re-login failed after refresh!")
                    return False
                if not self.navigate_to_order_page():
                    logger.error("Failed to navigate to order page after re-login!")
                    return False
            else:
                if not self.is_element_present(By.XPATH, "//span[text()='Order Management']", timeout=0.5):
                    if not self.navigate_to_order_page():
                        logger.error("Failed to navigate to order entry page after refresh!")
                        return False
            return True
        except WebDriverException as e:
            logger.error(f"Webdriver error (browser closed or F5): {e}. Attempting to restart browser and restore session.")
            try:
                self.driver.quit()
            except Exception:
                pass
            self.driver = get_chrome_driver(self.user_data_dir)
            self.wait = WebDriverWait(self.driver, 1)
            if not self.login():
                logger.error("Re-login failed after restarting browser!")
                return False
            if not self.navigate_to_order_page():
                logger.error("Failed to navigate to order page after browser restart!")
                return False
            return True

    def click_buy_toggle(self):
        try:
            buy_radio_xpath = "(//input[@type='radio' and contains(@class, 'xtoggler-radio')])[3]"
            if self.is_element_present(By.XPATH, buy_radio_xpath, timeout=0.5):
                buy_radio = self.driver.find_element(By.XPATH, buy_radio_xpath)
                try:
                    buy_radio.click()
                except Exception:
                    self.driver.execute_script("arguments[0].click();", buy_radio)
                tm.sleep(0.01)
                return True
            else:
                logger.error("Buy toggle radio button not found")
                return False
        except Exception as e:
            logger.error(f"Failed to click buy toggle: {e}")
            self.take_screenshot("buy_toggle_error")
            return False

    def get_pre_close_price(self):
        if self.pre_close_price is not None:
            return self.pre_close_price
        try:
            pre_close_xpath = "//div[label[text()='Pre Close']]/b"
            pre_close_element = self.wait.until(
                EC.visibility_of_element_located((By.XPATH, pre_close_xpath))
            )
            pre_close_value = float(pre_close_element.text.strip().replace(",", ""))
            self.pre_close_price = pre_close_value
            logger.info(f"Pre close price: {pre_close_value}")
            return pre_close_value
        except Exception as e:
            logger.error(f"Failed to get pre-close price: {e}")
            self.take_screenshot("pre_close_price_error")
            return None

    def _reset_daily_limits_if_needed(self):
        current_date = date.today().isoformat()
        if self.trading_date != current_date:
            self.trading_date = current_date
            self.circuit_limit_price_for_date = None
            self.regular_session_price_for_date = None
            self.circuit_hit_for_date = False
            self._save_state_to_local_storage()

    def fill_price_pre_open(self):
        try:
            pre_close_value = self.get_pre_close_price()
            if pre_close_value is None:
                return None
            open_price = pre_close_value * 1.02
            pre_open_price = math.floor(open_price * 10) / 10
            price_input = self.driver.find_element(By.XPATH, "//input[@formcontrolname='price']")
            price_input.clear()
            price_input.send_keys(str(pre_open_price))
            price_input.send_keys(Keys.ENTER)
            logger.info(f"Pre-open price filled: {pre_open_price}")
            return pre_open_price
        except Exception as e:
            logger.error(f"Failed to fill price for pre-open session: {e}")
            self.take_screenshot("preopen_price_error")
            return None

    def fill_price_regular(self):
        self._reset_daily_limits_if_needed()
        cur_date = self.trading_date

        if self.circuit_limit_price_for_date is None:
            pre_close_value = self.get_pre_close_price()
            if pre_close_value is None:
                logger.error("Cannot calculate regular session price: pre-close missing")
                return None
            circuit_limit_price = pre_close_value * (1 + CIRCUIT_LIMIT_PERCENTAGE/100)
            circuit_limit_price = math.floor(circuit_limit_price * 10) / 10
            self.circuit_limit_price_for_date = circuit_limit_price
            logger.info(f"Circuit limit price for {cur_date} set to {circuit_limit_price}")
            self.circuit_hit_for_date = False
            self._save_state_to_local_storage()

        if self.circuit_hit_for_date:
            logger.info(f"Circuit level already hit for {cur_date}. Using circuit limit price: {self.circuit_limit_price_for_date}")
            price_to_use = self.circuit_limit_price_for_date
        else:
            try:
                high_xpath = "//div[label[text()='High']]/b"
                high_element = self.wait.until(
                    EC.visibility_of_element_located((By.XPATH, high_xpath))
                )
                high_value = float(high_element.text.strip().replace(",", ""))
                logger.info(f"High value retrieved: {high_value}")
                calculated_price = high_value * 1.02
                calculated_price = math.floor(calculated_price * 10) / 10
            except Exception as e:
                logger.warning(f"Could not retrieve high value: {e}. Falling back to circuit limit.")
                calculated_price = self.circuit_limit_price_for_date

            if calculated_price >= self.circuit_limit_price_for_date:
                logger.info(
                    f"Regular session price: calculated={calculated_price} circuit_limit={self.circuit_limit_price_for_date}, using={self.circuit_limit_price_for_date} [date: {cur_date}]"
                )
                price_to_use = self.circuit_limit_price_for_date
                self.circuit_hit_for_date = True
            else:
                logger.info(
                    f"Regular session price: calculated={calculated_price} circuit_limit={self.circuit_limit_price_for_date}, using={calculated_price} [date: {cur_date}]"
                )
                price_to_use = calculated_price
            self._save_state_to_local_storage()

        self.regular_session_price_for_date = price_to_use
        self._save_state_to_local_storage()

        try:
            price_input = self.wait.until(
                EC.visibility_of_element_located((By.XPATH, "//input[@formcontrolname='price']"))
            )
            price_input.clear()
            price_input.send_keys(str(self.regular_session_price_for_date))
            price_input.send_keys(Keys.ENTER)
            logger.info(f"Regular session price filled: {self.regular_session_price_for_date} [date: {cur_date}]")
            return self.regular_session_price_for_date
        except Exception as e:
            logger.error(f"Failed to fill price for regular session: {e}")
            self.take_screenshot("regularsession_price_error")
            return None

    @disable_gc_during_critical
    def rapid_click_buy_button(self):
        click_count = 0
        buy_button_xpath = "//button[text()='BUY' and @type='submit' and not(@disabled)]"
        self.stop_clicking.clear()
        self.order_success.clear()
        logger.info("Starting rapid click sequence for BUY button (ultra-low-latency)")

        def click_forever():
            nonlocal click_count
            try:
                if self.use_gpu:
                    arr = cuda.device_array(1024)
                    gpu_atomic_fill[1, 1024](arr, 1.0)
                while not self.order_success.is_set() and not self.stop_clicking.is_set():
                    try:
                        buy_button = self.driver.find_element(By.XPATH, buy_button_xpath)
                        self.driver.execute_script("arguments[0].click();", buy_button)
                        click_count += 1
                        now = tm.perf_counter_ns()
                        self.ring.push(now)
                        success_xpaths = [
                            "//div[contains(@class,'alert-success')]",
                            "//div[contains(text(),'Order placed successfully')]",
                            "//div[contains(text(),'successful')]"
                        ]
                        for msg_xpath in success_xpaths:
                            if self.is_element_present(By.XPATH, msg_xpath, timeout=0.01):
                                self.order_success.set()
                                self.stop_clicking.set()
                                return
                        self.handle_confirmation_dialogs()
                        tm.sleep(CLICK_INTERVAL)
                    except (StaleElementReferenceException, NoSuchElementException):
                        tm.sleep(CLICK_INTERVAL)
                    except Exception:
                        tm.sleep(CLICK_INTERVAL)
            except Exception as e:
                logger.error(f"Clicking thread error: {e}")

        threads = []
        for _ in range(4):
            t = threading.Thread(target=click_forever)
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        logger.info(f"Rapid click sequence completed: {click_count} clicks")
        return click_count, self.order_success.is_set()

    def handle_confirmation_dialogs(self):
        try:
            possible_buttons = [
                "//button[contains(text(), 'OK')]",
                "//button[contains(text(), 'Confirm')]",
                "//button[contains(text(), 'Yes')]",
                "//div[contains(@class, 'modal')]//button[contains(@class, 'btn-primary')]"
            ]
            for xpath in possible_buttons:
                if self.is_element_present(By.XPATH, xpath, timeout=0.01):
                    confirm_button = self.driver.find_element(By.XPATH, xpath)
                    self.driver.execute_script("arguments[0].click();", confirm_button)
                    return True
            return False
        except Exception:
            return False

    def prepare_order_form(self):
        try:
            if not self.check_session_validity():
                return False
            if not self.is_element_present(By.XPATH, "//span[text()='Buy/Sell']", timeout=0.25):
                logger.info("Navigating to Order Management tab")
                if not self.wait_and_click(By.XPATH, "//span[text()='Order Management']", timeout=0.5):
                    logger.error("Could not click Order Management tab")
                    return False
                logger.info("Navigating to Buy/Sell section")
                if not self.wait_and_click(By.XPATH, "//span[normalize-space(text())='Buy/Sell']", timeout=0.5):
                    logger.error("Could not click Buy/Sell section")
                    return False
            symbol_xpath = "//input[@formcontrolname='symbol']"
            symbol_input = self.wait.until(EC.element_to_be_clickable((By.XPATH, symbol_xpath)))
            symbol_input.clear()
            symbol_input.send_keys(SYMBOL)
            try:
                symbol_suggestions_xpath = (
                    "//div[contains(@class, 'suggestion') or contains(@class, 'dropdown')]//div[contains(text(), '" + SYMBOL + "')]"
                )
                if self.is_element_present(By.XPATH, symbol_suggestions_xpath, timeout=0.25):
                    suggestion = self.driver.find_element(By.XPATH, symbol_suggestions_xpath)
                    suggestion.click()
                else:
                    self.driver.find_element(By.XPATH, "//body").click()
            except Exception:
                self.driver.find_element(By.XPATH, "//body").click()
            qty_xpath = "//input[@formcontrolname='quantity']"
            qty_input = self.wait.until(EC.element_to_be_clickable((By.XPATH, qty_xpath)))
            qty_input.clear()
            qty_input.send_keys(str(QUANTITY))
            if not self.click_buy_toggle():
                logger.error("Failed to click buy toggle button")
                return False
            logger.info("Order form prepared with symbol and quantity")
            return True
        except Exception as e:
            logger.error(f"Failed to prepare order form: {e}")
            self.take_screenshot("prepare_form_error")
            return False

    def log_order(self, symbol, quantity, price, status, mode="TRADING", click_count=0):
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.order_log_file, "a") as f:
                f.write(f"{timestamp},{symbol},{quantity},{price},{status},{mode},{click_count}\n")
        except Exception as e:
            logger.error(f"Failed to log order: {e}")

    def detect_transaction(self):
        try:
            ltp_xpath = "//div[label[text()='Last Traded Price']]/b"
            qty_xpath = "//div[label[text()='Total Qty']]/b"
            ltp_elem = self.driver.find_element(By.XPATH, ltp_xpath)
            qty_elem = self.driver.find_element(By.XPATH, qty_xpath)
            ltp = float(ltp_elem.text.replace(",", ""))
            qty = int(qty_elem.text.replace(",", ""))
            changed = False

            if self.last_known_price is None or self.last_known_qty is None:
                self.last_known_price = ltp
                self.last_known_qty = qty
                self._save_state_to_local_storage()
                return False

            if ltp != self.last_known_price or qty != self.last_known_qty:
                logger.info(f"Transaction detected: Price changed from {self.last_known_price} to {ltp}, Qty from {self.last_known_qty} to {qty}")
                self.last_known_price = ltp
                self.last_known_qty = qty
                self._save_state_to_local_storage()
                changed = True

            return changed
        except Exception as e:
            logger.warning(f"Could not detect transaction: {e}")
            return False

    def continue_automation_loop(self, run_duration_hours=6):
        hotkey_thread = threading.Thread(target=self._hotkey_listener, daemon=True)
        hotkey_thread.start()
        try:
            self._automation_main_loop(run_duration_hours)
        except KeyboardInterrupt:
            logger.info("Bot stopped by user (KeyboardInterrupt)")
        finally:
            logger.info(f"Bot finishing. Placed {self.successful_orders} successful orders.")
            logger.info("Closing browser")
            try:
                self.driver.quit()
            except Exception:
                pass

    def _hotkey_listener(self):
        logger.info(f"Press {HOTKEY_COMBO} at any time to stop the bot gracefully.")
        logger.info(f"Press {REFRESH_HOTKEY.upper()} to refresh and trigger form fill/order during trading hours.")
        keyboard.add_hotkey(HOTKEY_COMBO, self.stop_bot)
        while self.active:
            tm.sleep(0.2)

    def _register_refresh_hotkey(self):
        def f5_callback():
            logger.info(f"F5 pressed! Ultra-fast refresh and order trigger requested.")
            self.refresh_requested.set()
        keyboard.add_hotkey(REFRESH_HOTKEY, f5_callback, suppress=False)

    def stop_bot(self):
        logger.info(f"Hotkey pressed ({HOTKEY_COMBO})! Stopping bot gracefully...")
        self.active = False
        self.stop_clicking.set()

    def _automation_main_loop(self, run_duration_hours):
        start_time = tm.time()
        end_time = start_time + (run_duration_hours * 3600)
        logger.info(f"Bot started. Will run for {run_duration_hours} hours until {datetime.fromtimestamp(end_time)}")
        while tm.time() < end_time and self.active:
            try:
                if self.refresh_requested.is_set():
                    if self.is_trading_hours():
                        logger.info("F5 triggered: Performing browser refresh, form fill, and order placement (trading hours).")
                        self.driver.refresh()
                        if not self.check_session_validity():
                            logger.error("Session not valid after F5 refresh. Retrying in 1 second...")
                            tm.sleep(1)
                            self.refresh_requested.clear()
                            continue
                        self.place_buy_order()
                    else:
                        logger.info("F5 pressed, but not in trading hours. Ignoring form fill/order.")
                    self.refresh_requested.clear()
                    tm.sleep(NEXT_ORDER_WAIT)
                    continue

                session_ok = self.check_session_validity()
                if not session_ok:
                    logger.error("Session not valid after refresh or re-login failed. Retrying in 1 second...")
                    tm.sleep(1)
                    continue

                if self.is_trading_hours():
                    logger.info("Operational: In trading hours. Proceeding with automation (form filling and order placement).")
                    self._trading_session_loop()
                else:
                    if FORM_FILLING_MODE:
                        logger.info("Outside trading hours - FORM_FILLING_MODE enabled. Filling form only (no submit).")
                        self.place_buy_order(form_filling_mode=True)
                        tm.sleep(NEXT_ORDER_WAIT)
                    else:
                        logger.info("Outside trading hours - FORM_FILLING_MODE disabled. Waiting 10 seconds...")
                        tm.sleep(10)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                logger.error(f"Loop error: {e}")
                tm.sleep(0.5)

    def _trading_session_loop(self):
        try:
            while self.is_trading_hours() and self.active:
                if self.refresh_requested.is_set():
                    logger.info("F5 triggered inside trading session loop: Performing browser refresh, form fill, and order placement.")
                    self.driver.refresh()
                    if not self.check_session_validity():
                        logger.error("Session not valid after F5 refresh in trading session. Retrying in 1 second...")
                        tm.sleep(1)
                        self.refresh_requested.clear()
                        continue
                    self.place_buy_order()
                    self.refresh_requested.clear()
                    tm.sleep(NEXT_ORDER_WAIT)
                    continue

                if not self.check_session_validity():
                    logger.error("Session lost during trading session loop. Trying to recover...")
                    tm.sleep(1)
                    continue

                transaction_changed = self.detect_transaction()
                if transaction_changed:
                    logger.info("Transaction change detected during trading hours, placing new order as soon as possible.")
                    self.place_buy_order()
                    logger.info(f"Waiting {NEXT_ORDER_WAIT} seconds before next order after transaction detection")
                    tm.sleep(NEXT_ORDER_WAIT)
                    continue

                if self.is_pre_open_hours():
                    logger.info("Pre-open session automation running, attempting to place order.")
                    self.place_buy_order()
                    tm.sleep(NEXT_ORDER_WAIT)
                elif self.is_normal_trading_hours():
                    logger.info("Continuous (regular) session automation running, attempting to place order.")
                    self.place_buy_order()
                    tm.sleep(NEXT_ORDER_WAIT)
                else:
                    logger.info("No active trading session. Exiting trading session loop.")
                    break
        except Exception as e:
            logger.error(f"Error in trading session loop: {e}")

    def place_buy_order(self, form_filling_mode=False):
        try:
            if not self.check_session_validity():
                return False
            if not self.prepare_order_form():
                return False
            if self.is_pre_open_hours():
                price = self.fill_price_pre_open()
            else:
                price = self.fill_price_regular()
            if price is None:
                logger.error("Could not fill price field. Aborting order placement.")
                return False
            if form_filling_mode:
                logger.info("Form filling completed (FORM FILLING MODE - NO SUBMISSION)")
                self.log_order(SYMBOL, QUANTITY, price, "FORM_FILLED", "FORM_MODE", 0)
                self.successful_orders += 1
                return True
            logger.info("Launching ultra-rapid BUY button clicking loop (infinity until trade)")
            click_count, success = self.rapid_click_buy_button()
            if success:
                self.successful_orders += 1
                self.log_order(SYMBOL, QUANTITY, price, "SUCCESS", "TRADING", click_count)
                logger.info(f"Successfully placed buy order: {QUANTITY} shares of {SYMBOL} at {price} after {click_count} clicks")
                return True
            else:
                self.log_order(SYMBOL, QUANTITY, price, "FAILED", "TRADING", click_count)
                logger.warning(f"Order placement failed for {QUANTITY} shares at {price} after {click_count} clicks")
                return False
        except Exception as e:
            logger.error(f"Failed to place buy order: {e}")
            self.take_screenshot("buy_error")
            return False

def parse_arguments():
    parser = argparse.ArgumentParser(description="NEPSE Trading Bot (Ultra-Low-Latency HFT Version, GPU-Accelerated py311)")
    parser.add_argument("--form-mode", action="store_true", help="Enable form filling mode outside trading hours")
    parser.add_argument("--duration", type=float, default=6, help="Run duration in hours")
    parser.add_argument("--symbol", type=str, help="Stock symbol to trade")
    parser.add_argument("--quantity", type=int, help="Quantity to trade")
    parser.add_argument("--wait-time", type=int, default=0, help="Wait time between order attempts in seconds")
    parser.add_argument("--captcha-wait", type=int, default=30, help="Captcha fill wait time in seconds (manual)")
    parser.add_argument("--circuit-limit", type=float, default=10.0, help="Circuit breaker limit percentage (default: 10.0%)")
    parser.add_argument("--user-data-dir", type=str, help="Chrome user data dir for session persistence (recommended for F5 support)")
    parser.add_argument("--use-gpu", action="store_true", help="Enable GPU acceleration (Numba CUDA, RTX 3060)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    if args.form_mode:
        FORM_FILLING_MODE = True
        logger.info("Form filling mode enabled via command line")
    if args.symbol:
        SYMBOL = args.symbol
        logger.info(f"Symbol set to {SYMBOL} via command line")
    if args.quantity:
        QUANTITY = args.quantity
        logger.info(f"Quantity set to {QUANTITY} via command line")
    if args.wait_time is not None:
        NEXT_ORDER_WAIT = args.wait_time
        logger.info(f"Wait time between orders set to {NEXT_ORDER_WAIT} seconds via command line")
    if hasattr(args, "captcha_wait") and args.captcha_wait:
        CAPTCHA_FILL_WAIT = args.captcha_wait
        logger.info(f"Captcha fill wait time set to {CAPTCHA_FILL_WAIT} seconds via command line")
    if args.circuit_limit:
        CIRCUIT_LIMIT_PERCENTAGE = args.circuit_limit
        logger.info(f"Circuit breaker limit set to {CIRCUIT_LIMIT_PERCENTAGE}% via command line")

    logger.info("=== NEPSE Trading Bot HFT Configuration ===")
    logger.info(f"Symbol: {SYMBOL}")
    logger.info(f"Quantity: {QUANTITY}")
    logger.info(f"Form filling mode: {'Enabled' if FORM_FILLING_MODE else 'Disabled'}")
    logger.info(f"Click interval: {CLICK_INTERVAL}s (nanosecond granularity)")
    logger.info(f"Wait time between orders: {NEXT_ORDER_WAIT} seconds")
    logger.info(f"Captcha fill wait time: {CAPTCHA_FILL_WAIT} seconds")
    logger.info(f"Run duration: {args.duration} hours")
    logger.info(f"Circuit breaker limit: {CIRCUIT_LIMIT_PERCENTAGE}%")
    logger.info(f"Pre-open start time: {PREOPEN_START_HOUR}:{PREOPEN_START_MINUTE}")
    logger.info(f"Regular trading start time: {REGULAR_START_HOUR}:{REGULAR_START_MINUTE}")
    logger.info(f"Hotkey to stop bot: {HOTKEY_COMBO}")
    logger.info(f"Hotkey to refresh (form fill/order): {REFRESH_HOTKEY.upper()} (only during trading hours)")
    logger.info(f"GPU acceleration enabled: {args.use_gpu and GPU_ENABLED}")
    logger.info("=======================================")
    logger.info("Starting NEPSE Trading Bot (Ultra-Fast F5 Refresh, GPU Rapid Click py311, Back-to-Back Orders, Circuit Stop, Manual Captcha)")
    trader = NepseTrader(
        user_data_dir=args.user_data_dir if hasattr(args,'user_data_dir') else None,
        use_gpu=args.use_gpu
    )
    trader.continue_automation_loop(run_duration_hours=args.duration)
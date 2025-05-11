"""
NEPSE Trading Bot (Ultra-Low-Latency HFT Version, GPU-Accelerated for RTX 3060, Python 3.11)
- GPU-accelerated rapid buy clicking and rapid form filling (Numba CUDA).
- F5 (refresh) triggers order with minimal delay during trading hours.
- Back-to-back order placement: As soon as one order completes, immediately start the next during active trading hours.
- Stops placing new orders after a circuit level price order is placed.
- UPDATED: Auto-refresh and prepares order to submit at pre-open (10:30:00.000) and continuous (11:00:00.000) session boundaries with no delay.
- All long waits and loops are now interruptible by F5.
- Latency profiler measures boundary click-to-server-response with millisecond accuracy.
- Captcha must be filled manually by user if detected, with a blocking wait for human entry.
- Browser launches directly to login page (never blank).
- After every refresh, always navigates to the order entry page and waits for all fields before filling form.
- Robust error handling during form preparation, with logging and screenshots.
"""

import argparse
import logging
import os
import threading
import time as tm
from datetime import datetime, time, date, timedelta
import math
import sys
import gc

from numba import cuda
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

USERNAME = "GL302996"
PASSWORD = "Saanu####8775"
SYMBOL = "OMPL"
QUANTITY = 1000
FORM_FILLING_MODE = True
CLICK_INTERVAL = 1e-9
NEXT_ORDER_WAIT = 0
CAPTCHA_FILL_WAIT = 180  # Allow up to 3 minutes for manual captcha entry
PREOPEN_START_HOUR = 10
PREOPEN_START_MINUTE = 29
PREOPEN_BOUNDARY_HOUR = 10
PREOPEN_BOUNDARY_MINUTE = 30
PREOPEN_BOUNDARY_SECOND = 0
PREOPEN_END_HOUR = 10
PREOPEN_END_MINUTE = 45
REGULAR_START_HOUR = 10
REGULAR_START_MINUTE = 59
REGULAR_BOUNDARY_HOUR = 11
REGULAR_BOUNDARY_MINUTE = 0
REGULAR_BOUNDARY_SECOND = 0
REGULAR_END_HOUR = 15
REGULAR_END_MINUTE = 0
CIRCUIT_LIMIT_PERCENTAGE = 10

HOTKEY_COMBO = "ctrl+shift+q"
REFRESH_HOTKEY = "f5"
SHM_RING_SIZE = 65536

LATENCY_LOG_FILE = "latency_profiler.log"

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

class GPURingBuffer:
    def __init__(self, size=SHM_RING_SIZE):
        import numpy as np
        self.size = size
        self.buf = cuda.device_array(size, dtype=np.float64)
        self.head = cuda.to_device(np.array([0], dtype=np.int32))
        self.tail = cuda.to_device(np.array([0], dtype=np.int32))

    def push(self, value):
        import numpy as np
        head = self.head.copy_to_host()[0]
        tail = self.tail.copy_to_host()[0]
        next_head = (head + 1) % self.size
        if next_head == tail:
            return False
        cuda.to_device(np.array([value], dtype=np.float64), to=self.buf[head:head+1])
        self.head = cuda.to_device(np.array([next_head], dtype=np.int32))
        return True

    def pop(self):
        head = self.head.copy_to_host()[0]
        tail = self.tail.copy_to_host()[0]
        if head == tail:
            return None
        value = self.buf[tail].copy_to_host()
        next_tail = (tail + 1) % self.size
        self.tail = cuda.to_device(np.array([next_tail], dtype=np.int32))
        return value[()]

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
    driver.get("https://tms18.nepsetms.com.np/")  # Open login page immediately after browser launch
    return driver

class LatencyProfiler:
    def __init__(self, logfile=LATENCY_LOG_FILE):
        self.logfile = logfile
        self._reset()

    def _reset(self):
        self.boundary_order_time = None
        self.server_response_time = None

    def mark_boundary_click(self):
        self.boundary_order_time = tm.time_ns()

    def mark_server_response(self):
        self.server_response_time = tm.time_ns()

    def record(self, extra_info=""):
        if self.boundary_order_time and self.server_response_time:
            latency_ms = (self.server_response_time - self.boundary_order_time) / 1e6
            with open(self.logfile, "a") as f:
                log_str = f"{datetime.now().isoformat()},Latency(ms):{latency_ms:.3f},{extra_info}\n"
                f.write(log_str)
            logger.info(f"LatencyProfiler: {log_str.strip()}")
        self._reset()

class NepseTrader:
    def __init__(self, user_data_dir=None, use_gpu=True):
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
        self.last_known_high = None
        self.last_order_at_circuit = False

        self.use_gpu = use_gpu and cuda.is_available()
        if not self.use_gpu:
            logger.error("FATAL: GPU acceleration is required but CUDA is not available. Exiting.")
            sys.exit(1)
        logger.info("GPU acceleration enabled: True")
        self.ring = GPURingBuffer(SHM_RING_SIZE)
        self._restore_state_from_local_storage()
        self.refresh_requested = threading.Event()
        self._register_refresh_hotkey()
        self.browser_refresh_seconds = 1.7

        self.latency_profiler = LatencyProfiler()

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
                    if line.startswith("last_known_high:"):
                        val = line.strip().split(":", 1)[1].strip()
                        self.last_known_high = float(val) if val != "None" else None
                    if line.startswith("last_order_at_circuit:"):
                        val = line.strip().split(":", 1)[1].strip()
                        self.last_order_at_circuit = val == "True"
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
                f.write(f"last_known_high:{self.last_known_high}\n")
                f.write(f"last_order_at_circuit:{self.last_order_at_circuit}\n")
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
            return filename
        except Exception as e:
            logger.error(f"Screenshot error: {e}")
            return None

    def is_element_present(self, by, value, timeout=0.5):
        try:
            WebDriverWait(self.driver, timeout).until(EC.presence_of_element_located((by, value)))
            return True
        except (TimeoutException, NoSuchElementException):
            return False

    def wait_and_click(self, by, value, timeout=1, retries=1):
        for attempt in range(retries):
            if self.refresh_requested.is_set() or not self.active:
                return False
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
                self.interruptible_sleep(0.1)
        return False

    def interruptible_sleep(self, total_seconds):
        slept = 0
        interval = 0.2
        while slept < total_seconds:
            if self.refresh_requested.is_set() or not self.active:
                break
            tm.sleep(interval)
            slept += interval

    def is_pre_open_hours(self):
        now = datetime.now().time()
        pre_open_start = time(PREOPEN_START_HOUR, PREOPEN_START_MINUTE)
        pre_open_end = time(PREOPEN_END_HOUR, PREOPEN_END_MINUTE)
        return pre_open_start <= now <= pre_open_end

    def is_regular_trading_hours(self):
        now = datetime.now().time()
        normal_start = time(REGULAR_START_HOUR, REGULAR_START_MINUTE)
        normal_end = time(REGULAR_END_HOUR, REGULAR_END_MINUTE)
        return normal_start <= now <= normal_end

    def is_trading_hours(self):
        return self.is_pre_open_hours() or self.is_regular_trading_hours()

    def login(self):
        try:
            logger.info("Navigating to login page")
            self.driver.get(self.login_url)
            if self.is_element_present(By.XPATH, "//span[contains(text(), 'Dashboard')]"):
                logger.info("Already logged in")
                return True
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
            # Captcha detection and **guaranteed blocking wait** for human entry
            captcha_xpath = "//input[contains(@placeholder, 'Captcha') or @formcontrolname='captcha']"
            if self.is_element_present(By.XPATH, captcha_xpath, timeout=2):
                logger.info(
                    f"Captcha detected. Please fill the Captcha manually in the browser and submit the form."
                    f" Waiting for successful login (dashboard), up to {CAPTCHA_FILL_WAIT} seconds."
                )
                self.take_screenshot("captcha_required")
                start = tm.time()
                displayed_error = False
                while tm.time() - start < CAPTCHA_FILL_WAIT:
                    if self.is_element_present(By.XPATH, "//span[contains(text(), 'Dashboard')]", timeout=2):
                        logger.info("Captcha filled and login successful.")
                        return True
                    waited = int(tm.time() - start)
                    if waited % 10 == 0 and not displayed_error:
                        logger.info(f"Waited {waited} seconds for captcha entry...")
                        displayed_error = True
                    if waited % 10 != 0:
                        displayed_error = False
                    tm.sleep(2)
                logger.error("Timeout waiting for manual captcha entry. Please try again.")
                return False
            # Wait for dashboard (success after login)
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
                self.interruptible_sleep(0.01)
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
            self.last_known_high = None
            self.last_order_at_circuit = False
            self._save_state_to_local_storage()

    def fill_price_pre_open(self):
        try:
            pre_close_value = self.get_pre_close_price()
            if pre_close_value is None:
                return None, False
            open_price = pre_close_value * 1.02
            pre_open_price = math.floor(open_price * 10) / 10
            price_input = self.driver.find_element(By.XPATH, "//input[@formcontrolname='price']")
            price_input.clear()
            price_input.send_keys(str(pre_open_price))
            price_input.send_keys(Keys.ENTER)
            logger.info(f"Pre-open price filled: {pre_open_price}")
            return pre_open_price, False
        except Exception as e:
            logger.error(f"Failed to fill price for pre-open session: {e}")
            self.take_screenshot("preopen_price_error")
            return None, False

    def fill_price_regular(self):
        self._reset_daily_limits_if_needed()
        cur_date = self.trading_date

        if self.circuit_limit_price_for_date is None:
            pre_close_value = self.get_pre_close_price()
            if pre_close_value is None:
                logger.error("Cannot calculate regular session price: pre-close missing")
                return None, False
            circuit_limit_price = pre_close_value * (1 + CIRCUIT_LIMIT_PERCENTAGE/100)
            circuit_limit_price = math.floor(circuit_limit_price * 10) / 10
            self.circuit_limit_price_for_date = circuit_limit_price
            logger.info(f"Circuit limit price for {cur_date} set to {circuit_limit_price}")
            self.circuit_hit_for_date = False
            self.last_order_at_circuit = False
            self._save_state_to_local_storage()

        at_circuit = False
        if self.circuit_hit_for_date:
            logger.info(f"Circuit level already hit for {cur_date}. Using circuit limit price: {self.circuit_limit_price_for_date}")
            price_to_use = self.circuit_limit_price_for_date
            at_circuit = True
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
                at_circuit = True
            else:
                logger.info(
                    f"Regular session price: calculated={calculated_price} circuit_limit={self.circuit_limit_price_for_date}, using={calculated_price} [date: {cur_date}]"
                )
                price_to_use = calculated_price
                at_circuit = False
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
            return self.regular_session_price_for_date, at_circuit
        except Exception as e:
            logger.error(f"Failed to fill price for regular session: {e}")
            self.take_screenshot("regularsession_price_error")
            return None, at_circuit

    @disable_gc_during_critical
    def rapid_click_buy_button(self):
        import numpy as np
        from numba import cuda

        buy_button_xpath = "//button[text()='BUY' and @type='submit' and not(@disabled)]"
        MAX_CLICKS = 32768
        INTERVAL_NS = 250_000

        @cuda.jit
        def generate_click_times(start_time, interval_ns, click_times, n_clicks):
            i = cuda.grid(1)
            if i < n_clicks:
                click_times[i] = start_time + i * interval_ns

        def gpu_click_scheduler(n_clicks, interval_ns):
            threads_per_block = 256
            blocks = (n_clicks + threads_per_block - 1) // threads_per_block
            click_times = cuda.device_array(n_clicks, dtype=np.int64)
            start_time = np.int64(tm.time_ns())
            generate_click_times[blocks, threads_per_block](start_time, interval_ns, click_times, n_clicks)
            return click_times.copy_to_host()

        def busy_wait_until(target_ns, spin_threshold_ns=100_000):
            while True:
                if self.refresh_requested.is_set() or self.stop_clicking.is_set() or not self.active:
                    break
                now_ns = tm.time_ns()
                remaining_ns = target_ns - now_ns
                if remaining_ns <= 0:
                    break
                elif remaining_ns > spin_threshold_ns:
                    sleep_time = (remaining_ns - spin_threshold_ns) / 1e9
                    self.interruptible_sleep(sleep_time)
                else:
                    pass

        self.stop_clicking.clear()
        self.order_success.clear()
        logger.info("Starting enhanced GPU-accelerated rapid click sequence for BUY button")

        click_times = gpu_click_scheduler(MAX_CLICKS, INTERVAL_NS)
        click_count = 0

        profiler_boundary_triggered = False

        for target_ns in click_times:
            if self.order_success.is_set() or self.stop_clicking.is_set() or self.refresh_requested.is_set() or not self.active:
                break
            busy_wait_until(target_ns)
            if self.refresh_requested.is_set() or self.stop_clicking.is_set() or not self.active:
                break
            try:
                buy_button = self.driver.find_element(By.XPATH, buy_button_xpath)
                self.driver.execute_script("arguments[0].click();", buy_button)
                click_count += 1
                now = tm.perf_counter_ns()
                self.ring.push(now)
                if not profiler_boundary_triggered:
                    self.latency_profiler.mark_boundary_click()
                    profiler_boundary_triggered = True
                success_xpaths = [
                    "//div[contains(@class,'alert-success')]",
                    "//div[contains(text(),'Order placed successfully')]",
                    "//div[contains(text(),'successful')]"
                ]
                for msg_xpath in success_xpaths:
                    if self.is_element_present(By.XPATH, msg_xpath, timeout=0.01):
                        self.latency_profiler.mark_server_response()
                        self.latency_profiler.record(extra_info=f"OrderClicks:{click_count}")
                        self.order_success.set()
                        self.stop_clicking.set()
                        logger.info(f"Order success detected after {click_count} clicks.")
                        return click_count, True
                self.handle_confirmation_dialogs()
            except (StaleElementReferenceException, NoSuchElementException):
                continue
            except Exception as e:
                logger.warning(f"Click error: {e}")
                continue

        logger.info(f"Enhanced GPU-accelerated rapid click sequence completed: {click_count} clicks")
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
            logger.info("Navigating to order entry page for form preparation.")
            self.driver.get(self.url)
            # Step 1: Ensure logged in and page is order page
            if not self.is_element_present(By.XPATH, "//span[contains(text(), 'Dashboard')]", timeout=3):
                logger.warning("Not on dashboard after navigating to order entry page. Trying to login.")
                if not self.login():
                    logger.error("Login failed during form preparation.")
                    return False
                self.driver.get(self.url)
            # Step 2: Wait for Order Management tab
            try:
                order_mgmt_tab = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.XPATH, "//span[text()='Order Management']"))
                )
            except TimeoutException:
                logger.error("Order Management tab not found after 10s. Possibly not logged in or wrong page. Taking screenshot.")
                self.take_screenshot("order_mgmt_tab_missing")
                return False
            # Step 3: Buy/Sell tab logic
            try:
                if not self.is_element_present(By.XPATH, "//span[text()='Buy/Sell']", timeout=2):
                    logger.info("Buy/Sell tab not visible, switching...")
                    if not self.wait_and_click(By.XPATH, "//span[text()='Order Management']", timeout=3):
                        logger.error("Could not click Order Management tab.")
                        self.take_screenshot("cant_click_order_mgmt")
                        return False
                    if not self.wait_and_click(By.XPATH, "//span[normalize-space(text())='Buy/Sell']", timeout=3):
                        logger.error("Could not click Buy/Sell section.")
                        self.take_screenshot("cant_click_buysell")
                        return False
            except Exception as e:
                logger.error(f"Exception while switching tabs: {e}")
                self.take_screenshot("tab_switch_exception")
                return False
            # Step 4: Wait for all fields
            required_fields = [
                "//input[@formcontrolname='symbol']",
                "//input[@formcontrolname='quantity']",
                "//input[@formcontrolname='price']"
            ]
            for field_xpath in required_fields:
                try:
                    self.wait.until(EC.visibility_of_element_located((By.XPATH, field_xpath)))
                except TimeoutException:
                    logger.error(f"Order form field not found or not visible: {field_xpath}")
                    self.take_screenshot("form_field_missing")
                    return False
            logger.info("Order form is ready for filling!")
            return True
        except Exception as e:
            logger.error(f"Error preparing order form: {e}")
            self.take_screenshot("prepare_form_error")
            return False

    def log_order(self, symbol, quantity, price, status, mode="TRADING", click_count=0):
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.order_log_file, "a") as f:
                f.write(f"{timestamp},{symbol},{quantity},{price},{status},{mode},{click_count}\n")
        except Exception as e:
            logger.error(f"Failed to log order: {e}")

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
        self.refresh_requested.set()

    def _wait_until(self, target_dt):
        while (now := datetime.now()) < target_dt:
            delta = (target_dt - now).total_seconds()
            interval = min(0.2, delta)
            if self.refresh_requested.is_set() or not self.active or delta <= 0:
                break
            tm.sleep(interval)

    def _automation_main_loop(self, run_duration_hours):
        start_time = tm.time()
        end_time = start_time + (run_duration_hours * 3600)
        logger.info(f"Bot started. Will run for {run_duration_hours} hours until {datetime.fromtimestamp(end_time)}")
        while tm.time() < end_time and self.active:
            try:
                now = datetime.now()
                preopen_boundary = now.replace(hour=PREOPEN_BOUNDARY_HOUR, minute=PREOPEN_BOUNDARY_MINUTE, second=PREOPEN_BOUNDARY_SECOND, microsecond=0)
                if now > preopen_boundary:
                    preopen_boundary += timedelta(days=1)
                reg_boundary = now.replace(hour=REGULAR_BOUNDARY_HOUR, minute=REGULAR_BOUNDARY_MINUTE, second=REGULAR_BOUNDARY_SECOND, microsecond=0)
                if now > reg_boundary:
                    reg_boundary += timedelta(days=1)
                if self.is_pre_open_hours():
                    logger.info("In pre-open session. Preparing for atomic order at 10:30:00.000 boundary.")
                    self._schedule_boundary_order(
                        boundary_time=preopen_boundary,
                        fill_price_fn=self.fill_price_pre_open,
                        boundary_type="PREOPEN"
                    )
                    continue
                if self.is_regular_trading_hours() and now < reg_boundary:
                    logger.info("In pre-regular session. Preparing for atomic order at 11:00:00.000 boundary.")
                    self._schedule_boundary_order(
                        boundary_time=reg_boundary,
                        fill_price_fn=self.fill_price_regular,
                        boundary_type="REGULAR"
                    )
                    continue
                if self.refresh_requested.is_set():
                    if self.is_trading_hours():
                        logger.info("F5 triggered: Performing browser refresh, form fill, and order placement (trading hours).")
                        self.driver.refresh()
                        if not self.check_session_validity():
                            logger.error("Session not valid after F5 refresh. Retrying in 1 second...")
                            self.interruptible_sleep(1)
                            self.refresh_requested.clear()
                            continue
                        self.place_buy_order()
                    else:
                        logger.info("F5 pressed, but not in trading hours. Ignoring form fill/order.")
                    self.refresh_requested.clear()
                    continue
                session_ok = self.check_session_validity()
                if not session_ok:
                    logger.error("Session not valid after refresh or re-login failed. Retrying in 1 second...")
                    self.interruptible_sleep(1)
                    continue
                if self.is_trading_hours():
                    logger.info("Operational: In trading hours. Proceeding with back-to-back automation (continuous order placement).")
                    self._back_to_back_trading_loop()
                else:
                    if FORM_FILLING_MODE:
                        logger.info("Outside trading hours - FORM_FILLING_MODE enabled. Filling form only (no submit).")
                        self.place_buy_order(form_filling_mode=True)
                    else:
                        logger.info("Outside trading hours - FORM_FILLING_MODE disabled. Waiting 10 seconds...")
                        self.interruptible_sleep(10)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                logger.error(f"Loop error: {e}")
                self.interruptible_sleep(0.5)

    def _schedule_boundary_order(self, boundary_time, fill_price_fn, boundary_type):
        now = datetime.now()
        refresh_margin = timedelta(seconds=self.browser_refresh_seconds + 0.5)  # Added more margin to avoid being too fast
        refresh_time = boundary_time - refresh_margin
        logger.info(f"{boundary_type} session boundary at {boundary_time}. Auto-refresh scheduled at {refresh_time} (will measure & adjust).")
        self._wait_until(refresh_time)
        if not self.active:
            return
        logger.info(f"Auto-refreshing at {datetime.now()} for {boundary_type} boundary order.")
        self.driver.refresh()
        if not self.check_session_validity():
            logger.error("Session not valid after auto-refresh! Retrying in 2 seconds...")
            self.interruptible_sleep(2)
            return
        if not self.prepare_order_form():
            logger.error("Failed to prepare order form before boundary order.")
            return
        price, at_circuit = fill_price_fn()
        logger.info(f"Order form ready at {datetime.now()} for {boundary_type} boundary order. Waiting for boundary...")
        while datetime.now() < boundary_time:
            delta = (boundary_time - datetime.now()).total_seconds()
            if self.refresh_requested.is_set() or not self.active or delta <= 0:
                return
            tm.sleep(min(0.1, delta))
        logger.info(f"Triggering rapid BUY at {datetime.now()} (should be nearly atomic with boundary).")
        self.latency_profiler._reset()
        click_count, success = self.rapid_click_buy_button()
        if success:
            self.successful_orders += 1
            self.log_order(SYMBOL, QUANTITY, price, "SUCCESS", "BOUNDARY", click_count)
            self.last_order_at_circuit = at_circuit
            self._save_state_to_local_storage()
            logger.info(f"Successfully placed boundary buy order: {QUANTITY} shares of {SYMBOL} at {price} after {click_count} clicks")
        else:
            self.log_order(SYMBOL, QUANTITY, price, "FAILED", "BOUNDARY", click_count)
            logger.warning(f"Boundary order placement failed for {QUANTITY} shares at {price} after {click_count} clicks")

    def _back_to_back_trading_loop(self):
        while self.is_trading_hours() and self.active:
            if self.refresh_requested.is_set():
                logger.info("F5 triggered inside trading session loop: Performing browser refresh, form fill, and order placement.")
                self.driver.refresh()
                if not self.check_session_validity():
                    logger.error("Session not valid after F5 refresh in trading session. Retrying in 1 second...")
                    self.interruptible_sleep(1)
                    self.refresh_requested.clear()
                    continue
                self.place_buy_order()
                self.refresh_requested.clear()
                continue
            if not self.check_session_validity():
                logger.error("Session lost during trading session loop. Trying to recover...")
                self.interruptible_sleep(1)
                continue
            if self.last_order_at_circuit:
                logger.info("Circuit level price order was placed. Stopping further back-to-back orders for this session.")
                break
            logger.info("Placing next order immediately (back-to-back mode).")
            at_circuit = self.place_buy_order()
            if at_circuit:
                logger.info("Circuit level price order just placed. Stopping order loop.")
                break

    def place_buy_order(self, form_filling_mode=False):
        try:
            if self.refresh_requested.is_set() or not self.active:
                return False
            if not self.check_session_validity():
                return False
            if not self.prepare_order_form():
                return False
            if self.is_pre_open_hours():
                price, at_circuit = self.fill_price_pre_open()
            else:
                price, at_circuit = self.fill_price_regular()
            if price is None:
                logger.error("Could not fill price field. Aborting order placement.")
                return False
            if form_filling_mode:
                logger.info("Form filling completed (FORM FILLING MODE - NO SUBMISSION)")
                self.log_order(SYMBOL, QUANTITY, price, "FORM_FILLED", "FORM_MODE", 0)
                self.successful_orders += 1
                return False
            logger.info("Launching GPU-accelerated ultra-rapid BUY button clicking loop (infinity until trade)")
            click_count, success = self.rapid_click_buy_button()
            if success:
                self.successful_orders += 1
                self.log_order(SYMBOL, QUANTITY, price, "SUCCESS", "TRADING", click_count)
                self.last_order_at_circuit = at_circuit
                self._save_state_to_local_storage()
                logger.info(f"Successfully placed buy order: {QUANTITY} shares of {SYMBOL} at {price} after {click_count} clicks")
                return at_circuit
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
    parser.add_argument("--captcha-wait", type=int, default=180, help="Captcha fill wait time in seconds (manual)")
    parser.add_argument("--circuit-limit", type=float, default=10.0, help="Circuit breaker limit percentage (default: 10.0%)")
    parser.add_argument("--user-data-dir", type=str, help="Chrome user data dir for session persistence (recommended for F5 support)")
    parser.add_argument("--use-gpu", action="store_true", default=True, help="Enable GPU acceleration (Numba CUDA, RTX 3060)")
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
    logger.info(f"Captcha fill wait time: {CAPTCHA_FILL_WAIT} seconds (manual)")
    logger.info(f"Run duration: {args.duration} hours")
    logger.info(f"Circuit breaker limit: {CIRCUIT_LIMIT_PERCENTAGE}%")
    logger.info(f"Pre-open start time: {PREOPEN_START_HOUR}:{PREOPEN_START_MINUTE}")
    logger.info(f"Regular trading start time: {REGULAR_START_HOUR}:{REGULAR_START_MINUTE}")
    logger.info(f"Hotkey to stop bot: {HOTKEY_COMBO}")
    logger.info(f"Hotkey to refresh (form fill/order): {REFRESH_HOTKEY.upper()} (only during trading hours)")
    logger.info(f"GPU acceleration enabled: {args.use_gpu and cuda.is_available()}")
    logger.info("=======================================")
    logger.info("Starting NEPSE Trading Bot (Ultra-Fast F5 Refresh, GPU Rapid Click py311, Back-to-Back Orders, Circuit Stop, Atomic Boundary Orders, Latency Profiler, Manual Captcha)")
    trader = NepseTrader(
        user_data_dir=args.user_data_dir if hasattr(args,'user_data_dir') else None,
        use_gpu=args.use_gpu
    )
    trader.continue_automation_loop(run_duration_hours=args.duration)
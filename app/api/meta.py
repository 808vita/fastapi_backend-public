from typing import Dict
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from ..schemas import MetaAdsRequest, ScreenshotRequest
from ..utils import get_meta_ads, get_full_page_screenshot
import base64
from playwright.async_api import async_playwright
import os
from dotenv import load_dotenv
import logging
import asyncio
import json

load_dotenv()

router = APIRouter()


@router.post("/meta-ads", response_model=dict)
async def meta_ads(request: MetaAdsRequest):
    """Fetches ads data from Meta Ad Library API"""
    try:
        if not any(request.__dict__.values()):
            raise HTTPException(
                status_code=400, detail="Please provide at least one query parameter."
            )
        result = await get_meta_ads(**request.model_dump(exclude_none=True))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/screenshot", response_model=dict)
async def screenshot(request: ScreenshotRequest):
    """Takes a full page screenshot of the URL"""
    try:
        result = await get_full_page_screenshot(request.url)
        return {"base64_string": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


FACEBOOK_USERNAME = os.getenv("FACEBOOK_USERNAME")
FACEBOOK_PASSWORD = os.getenv("FACEBOOK_PASSWORD")

if not FACEBOOK_USERNAME or not FACEBOOK_PASSWORD:
    raise ValueError("Facebook credentials are not set in .env file")


# # Configure logging
# logging.basicConfig(
#     level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
# )


async def capture_screenshot_base64(page, step_name) -> str:
    """Captures a screenshot and returns it as a base64 encoded string."""
    logging.debug(f"Capturing screenshot for: {step_name}")
    screenshot_bytes = await page.screenshot()
    base64_string = base64.b64encode(screenshot_bytes).decode("utf-8")
    return base64_string


@router.post("/meta-login", response_model=dict)
async def login_to_facebook():
    screenshot_dict: Dict[str, str] = {}
    try:
        logging.debug("Starting login process")
        async with async_playwright() as p:
            logging.debug("Launching browser")
            browser = await p.chromium.launch()
            logging.debug("Creating new page")
            page = await browser.new_page()
            logging.debug("Navigating to Facebook")
            await page.goto("https://www.facebook.com/")

            # Capture screenshot of login page
            screenshot_dict["login_page"] = await capture_screenshot_base64(
                page, "login_page"
            )

            # Locate elements
            logging.debug("Locating elements")
            email_input = page.locator("#email")
            password_input = page.locator("#pass")
            login_button = page.locator('button[name="login"]')

            if not (
                await email_input.count() > 0
                and await password_input.count() > 0
                and await login_button.count() > 0
            ):
                logging.debug("Login elements not found")
                await browser.close()
                raise HTTPException(
                    status_code=500, detail="Login elements not found on the page"
                )

            # Enter credentials
            logging.debug("Filling credentials")
            await email_input.fill(FACEBOOK_USERNAME)
            await password_input.fill(FACEBOOK_PASSWORD)

            # Capture screenshot after filling credentials
            screenshot_dict["credentials_filled"] = await capture_screenshot_base64(
                page, "credentials_filled"
            )

            # Submit
            logging.debug("Clicking login button")
            await login_button.click()

            # Check for successful login
            logging.debug("Waiting for load state: networkidle")
            await page.wait_for_load_state(state="networkidle")

            current_url = page.url
            logging.debug(f"Current URL after load: {current_url}")

            if "facebook.com/" in current_url:  # Check for any facebook page

                # Re-locate login elements
                email_input_after_redirect = page.locator("#email")
                password_input_after_redirect = page.locator("#pass")
                login_button_after_redirect = page.locator('button[name="login"]')

                if not (
                    await email_input_after_redirect.count() > 0
                    and await password_input_after_redirect.count() > 0
                    and await login_button_after_redirect.count() > 0
                ):
                    logging.debug("Successful Login, login elements not found")
                    screenshot_dict["home_page"] = await capture_screenshot_base64(
                        page, "home_page"
                    )
                    await browser.close()
                    return {
                        "status": "success",
                        "message": "Logged in successfully!",
                        "screenshots": screenshot_dict,
                    }
                else:
                    logging.debug("Login still failed, login elements still present")
                    screenshot_dict["error_page"] = await capture_screenshot_base64(
                        page, "error_page"
                    )
                    await browser.close()
                    return {
                        "status": "error",
                        "message": "Login failed or redirected elsewhere, check screenshots",
                        "screenshots": screenshot_dict,
                    }

            else:
                logging.debug("Login failed or redirected elsewhere")
                screenshot_dict["error_page"] = await capture_screenshot_base64(
                    page, "error_page"
                )
                await browser.close()
                return {
                    "status": "error",
                    "message": "Login failed or redirected elsewhere, check screenshots",
                    "screenshots": screenshot_dict,
                }

    except Exception as e:
        logging.error(f"Error during login: {e}", exc_info=True)
        screenshot_dict["error_page"] = await capture_screenshot_base64(
            page, "error_page"
        )
        await browser.close()
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during login: {e}",
            headers={"screenshots": str(screenshot_dict)},
        )


@router.websocket("/meta-login-ws")
async def websocket_login(websocket: WebSocket):
    await websocket.accept()
    screenshot_dict: Dict[str, str] = {}
    browser = None
    try:
        logging.debug("Starting login process")
        async with async_playwright() as p:
            logging.debug("Launching browser")
            browser = await p.chromium.launch()
            logging.debug("Creating new page")
            page = await browser.new_page()
            logging.debug("Navigating to Facebook")
            await page.goto("https://www.facebook.com/")

            # Capture screenshot of login page
            screenshot_dict["login_page"] = await capture_screenshot_base64(
                page, "login_page"
            )
            await websocket.send_json(
                {
                    "type": "screenshot",
                    "step": "login_page",
                    "image": screenshot_dict["login_page"],
                }
            )

            # Locate elements
            logging.debug("Locating elements")
            email_input = page.locator("#email")
            password_input = page.locator("#pass")
            login_button = page.locator('button[name="login"]')

            if not (
                await email_input.count() > 0
                and await password_input.count() > 0
                and await login_button.count() > 0
            ):
                logging.debug("Login elements not found")
                await browser.close()
                await websocket.send_json(
                    {"type": "error", "message": "Login elements not found on the page"}
                )
                return

            # Enter credentials
            logging.debug("Filling credentials")
            await email_input.fill(FACEBOOK_USERNAME)
            await password_input.fill(FACEBOOK_PASSWORD)

            # Capture screenshot after filling credentials
            screenshot_dict["credentials_filled"] = await capture_screenshot_base64(
                page, "credentials_filled"
            )
            await websocket.send_json(
                {
                    "type": "screenshot",
                    "step": "credentials_filled",
                    "image": screenshot_dict["credentials_filled"],
                }
            )

            # Submit
            logging.debug("Clicking login button")
            await login_button.click()

            # Check for successful login
            logging.debug("Waiting for load state: networkidle")
            await page.wait_for_load_state(state="networkidle")

            current_url = page.url
            logging.debug(f"Current URL after load: ")

            while True:

                if "facebook.com/two_step_verification/authentication" in current_url:
                    logging.debug("Captcha page detected")

                    captcha_input = page.locator("input[type='text']")
                    captcha_image = page.locator("img[src*='captcha/tfbimage']")

                    try:
                        continue_button = page.get_by_role("button", name="தொடர்க")
                    except:
                        try:
                            continue_button = page.get_by_role(
                                "button", name="Continue"
                            )
                        except:
                            await websocket.send_json(
                                {
                                    "type": "error",
                                    "message": "captcha continue button not found",
                                }
                            )
                            await browser.close()
                            return

                    if not (
                        await captcha_input.count() > 0
                        and await captcha_image.count() > 0
                    ):
                        await websocket.send_json(
                            {"type": "error", "message": "captcha elements not found"}
                        )
                        await browser.close()
                        return

                    screenshot_dict["captcha_page"] = await capture_screenshot_base64(
                        page, "captcha_page"
                    )

                    await websocket.send_json(
                        {
                            "type": "captcha",
                            "image": screenshot_dict["captcha_page"],
                            "message": "Enter captcha code",
                        }
                    )

                    data = await websocket.receive_json()
                    if not data:
                        await websocket.send_json(
                            {"type": "error", "message": "No data received for captcha"}
                        )
                        await browser.close()
                        return

                    if isinstance(data, dict) and "code" in data:
                        logging.debug("received captcha code")
                        await captcha_input.fill(data["code"])
                        await continue_button.click()
                        await page.wait_for_load_state(state="networkidle")
                        current_url = page.url
                        logging.debug(f"Current URL after captcha: ")
                        screenshot_dict["after_captcha_submitted"] = (
                            await capture_screenshot_base64(
                                page, "after_captcha_submitted"
                            )
                        )
                        await websocket.send_json(
                            {
                                "type": "screenshot",
                                "step": "after_captcha_submitted",
                                "image": screenshot_dict["after_captcha_submitted"],
                            }
                        )

                    else:
                        await websocket.send_json(
                            {"type": "error", "message": "invalid captcha response"}
                        )
                        await browser.close()
                        return

                elif "facebook.com/two_step_verification/two_factor/" in current_url:

                    logging.debug("Two factor authenticator code detected")

                    auth_input = page.locator("input[type='text']")

                    try:
                        continue_button = page.get_by_role("button", name="தொடர்க")
                    except:
                        try:
                            continue_button = page.get_by_role(
                                "button", name="Continue"
                            )
                        except:
                            await websocket.send_json(
                                {
                                    "type": "error",
                                    "message": "2fa continue button not found",
                                }
                            )
                            await browser.close()
                            return

                    if not (await auth_input.count() > 0):
                        await websocket.send_json(
                            {"type": "error", "message": "2fa input not found"}
                        )
                        await browser.close()
                        return

                    screenshot_dict["2fa_page"] = await capture_screenshot_base64(
                        page, "2fa_page"
                    )
                    await websocket.send_json(
                        {
                            "type": "2fa",
                            "image": screenshot_dict["2fa_page"],
                            "message": "Enter 2FA Code",
                        }
                    )

                    data = await websocket.receive_json()
                    if not data:
                        await websocket.send_json(
                            {"type": "error", "message": "No data received for auth"}
                        )
                        await browser.close()
                        return

                    if isinstance(data, dict) and "code" in data:
                        logging.debug("received auth code")
                        await auth_input.fill(data["code"])
                        await continue_button.click()
                        await page.wait_for_load_state(state="networkidle")
                        current_url = page.url
                        logging.debug(f"Current URL after auth code: ")
                        screenshot_dict["after_2fa_submitted"] = (
                            await capture_screenshot_base64(page, "after_2fa_submitted")
                        )
                        await websocket.send_json(
                            {
                                "type": "screenshot",
                                "step": "after_2fa_submitted",
                                "image": screenshot_dict["after_2fa_submitted"],
                            }
                        )

                    else:
                        await websocket.send_json(
                            {"type": "error", "message": "invalid auth response"}
                        )
                        await browser.close()
                        return

                else:
                    # Re-locate login elements
                    email_input_after_redirect = page.locator("#email")
                    password_input_after_redirect = page.locator("#pass")
                    login_button_after_redirect = page.locator('button[name="login"]')

                    if not (
                        await email_input_after_redirect.count() > 0
                        and await password_input_after_redirect.count() > 0
                        and await login_button_after_redirect.count() > 0
                    ):
                        logging.debug("Successful Login, login elements not found")
                        screenshot_dict["home_page"] = await capture_screenshot_base64(
                            page, "home_page"
                        )
                        await websocket.send_json(
                            {
                                "type": "success",
                                "message": "Logged in successfully!",
                                "screenshots": screenshot_dict,
                            }
                        )
                        await browser.close()
                        return
                    else:
                        logging.debug(
                            "Login still failed, login elements still present"
                        )
                        screenshot_dict["error_page"] = await capture_screenshot_base64(
                            page, "error_page"
                        )
                        await websocket.send_json(
                            {
                                "type": "error",
                                "message": "Login failed or redirected elsewhere, check screenshots",
                                "screenshots": screenshot_dict,
                            }
                        )
                        await browser.close()
                        return

    except Exception as e:
        logging.error(f"Error during login: {e}", exc_info=True)
        if "page" in locals() and page:
            screenshot_dict["error_page"] = await capture_screenshot_base64(
                page, "error_page"
            )
        await websocket.send_json(
            {
                "type": "error",
                "message": f"An error occurred during login: {e}",
                "screenshots": screenshot_dict,
            }
        )
    finally:
        if browser:
            await browser.close()

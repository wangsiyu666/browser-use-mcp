import os
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import BrowserContextConfig, BrowserContextWindowSize
from browser_use.agent.service import Agent
from utils.agent_state import AgentState
from utils.utils import get_latest_files
from langchain_openai.chat_models import ChatOpenAI

_global_browser = None
_global_browser_context = None
_global_agent = None


async def run_org_agent(
        llm,
        use_own_browser,
        keep_browser_open,
        headless,
        disable_security,
        window_w,
        window_h,
        save_recording_path,
        save_agent_history_path,
        save_trace_path,
        task,
        max_steps,
        use_vision,
        max_actions_per_step,
        tool_calling_method,
        chrome_cdp,
        max_input_tokens
):
    try:
        global _global_browser, _global_browser_context, _global_agent

        extra_chromium_args = [f"--window-size={window_w},{window_h}"]
        cdp_url = chrome_cdp

        if use_own_browser:
            cdp_url = os.getenv("CHROME_CDP", chrome_cdp)
            chrome_path = os.getenv("CHROME_PATH", None)
            if chrome_path == "":
                chrome_path = None
            chrome_use_path = os.getenv("CHROME_USER_DATA", None)
            if chrome_use_path:
                extra_chromium_args += [f"--user-data-dir={chrome_use_path}"]

        else:
            chrome_path = None
        if _global_browser is None:
            _global_browser = Browser(
                config=BrowserConfig(
                    headless=headless,
                    cdp_url=cdp_url,
                    disable_security=disable_security,
                    chrome_instance_path=chrome_path,
                    extra_chromium_args=extra_chromium_args,
                )
            )
        if _global_browser_context is None:
            _global_browser_context = await _global_browser.new_context(
                config=BrowserContextConfig(
                    trace_path=save_trace_path if save_trace_path else None,
                    save_recording_path=(save_recording_path if save_recording_path else None),
                    no_viewport=False,
                    browser_window_size=BrowserContextWindowSize(width=window_w, height=window_h)
                )
            )

        if _global_agent is None:
            _global_agent = Agent(
                task=task,
                llm=llm,
                use_vision=use_vision,
                browser=_global_browser,
                browser_context=_global_browser_context,
                max_actions_per_step=max_actions_per_step,
                tool_calling_method=tool_calling_method,
                max_input_tokens=max_input_tokens,
                generate_gif=True
            )
        history = await _global_agent.run(max_steps=max_steps)

        history_file = os.path.join(save_agent_history_path, f"{_global_agent.state.agent_id}.json")
        _global_agent.save_history(history_file)
        final_result = history.final_result()
        errors = history.errors()
        model_actions = history.model_actions()
        model_thoughts = history.model_actions()

        trace_file = get_latest_files(save_trace_path)

        return (
            final_result,
            errors,
            model_actions,
            model_thoughts,
            trace_file.get(".zip"),
            history_file,
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        errors = str(e) + "\n" + traceback.format_exc()
        return "", errors, "", "", None, None

    finally:
        _global_agent = None

        if not keep_browser_open:
            if _global_browser_context:
                await _global_browser_context.close()
                _global_browser_context = None

            if _global_browser:
                await _global_browser.close()
                _global_browser = None


async def run_browser_agent(
        agent_type,
        # llm_provider,
        llm_model_name,
        # llm_num_ctx,
        # llm_temperature,
        llm_base_url,
        llm_api_key,
        use_own_browser,
        keep_browser_open,
        headless,
        disable_security,
        window_w,
        window_h,
        save_recoding_path,
        save_agent_history_path,
        save_trace_path,
        enable_recording,
        task,
        # add_infos,
        max_steps,
        use_vision,
        max_actions_per_step,
        tool_calling_method,
        chrome_cdp,
        max_input_tokens,
):
    try:
        if not enable_recording:
            save_recoding_path = None

        if save_recoding_path:
            os.makedirs(save_recoding_path, exist_ok=True)

        llm = ChatOpenAI(
            model=llm_model_name,
            base_url=llm_base_url,
            api_key=llm_api_key
        )
        if agent_type == "org":
            (
                final_result,
                errors,
                model_actions,
                model_thoughts,
                trace_file,
                history_file
            ) = await run_org_agent(
                llm=llm,
                use_own_browser=use_own_browser,
                keep_browser_open=keep_browser_open,
                headless=headless,
                disable_security=disable_security,
                window_w=window_w,
                window_h=window_h,
                save_recording_path=save_recoding_path,
                save_agent_history_path=save_agent_history_path,
                save_trace_path=save_trace_path,
                task=task,
                max_steps=max_steps,
                use_vision=use_vision,
                max_actions_per_step=max_actions_per_step,
                tool_calling_method=tool_calling_method,
                chrome_cdp=chrome_cdp,
                max_input_tokens=max_input_tokens
            )
        elif agent_type == "custom":
            pass
        else:
            raise ValueError(f"Invalid agent type: {agent_type}")

        gif_path = os.path.join(os.path.dirname(__file__), "agent_history.gif")

        return (
            final_result,
            errors,
            model_actions,
            model_thoughts,
            gif_path,
            trace_file,
            history_file,
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        errors = str(e) + "\n" + traceback.format_exc()
        return (
            "",
            errors,
            "",
            "",
            None,
            None,
            None,
        )

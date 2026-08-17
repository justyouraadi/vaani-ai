from vaani.bargein.manager import BargeinManager


def test_interrupt_when_agent_active():
    bm = BargeinManager()
    bm.set_agent_active(True)
    assert bm.notify_user_speech() is True
    assert bm.interrupted is True
    assert bm.set_agent_active(False) is None


def test_no_interrupt_when_agent_idle():
    bm = BargeinManager()
    assert bm.notify_user_speech() is False
    assert bm.interrupted is False


def test_second_hint_while_interrupted_ignored():
    bm = BargeinManager()
    bm.set_agent_active(True)
    assert bm.notify_user_speech() is True
    assert bm.notify_user_speech() is False


def test_clear_resets_interrupt():
    bm = BargeinManager()
    bm.set_agent_active(True)
    bm.notify_user_speech()
    bm.clear()
    assert bm.interrupted is False


def test_talking_to_idle_then_agent_returns():
    bm = BargeinManager()
    bm.set_agent_active(True)
    bm.notify_user_speech()
    bm.clear()
    bm.set_agent_active(True)
    assert bm.notify_user_speech() is True
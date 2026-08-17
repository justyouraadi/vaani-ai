class BargeinManager:
    def __init__(self):
        self._agent_active = False
        self._interrupted = False

    def set_agent_active(self, active: bool) -> None:
        self._agent_active = active
        if not active:
            self._interrupted = False

    def notify_user_speech(self) -> bool:
        if self._agent_active and not self._interrupted:
            self._interrupted = True
            self._agent_active = False
            return True
        return False

    def clear(self) -> None:
        self._interrupted = False

    @property
    def interrupted(self) -> bool:
        return self._interrupted
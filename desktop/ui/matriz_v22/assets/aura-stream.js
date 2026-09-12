(function (global) {
  "use strict";

  function AuraStream() {}

  AuraStream.start = function (opts) {
    opts = opts || {};
    var url = opts.url || "";
    var pollUrl = opts.pollUrl || "";
    var onState = typeof opts.onState === "function" ? opts.onState : function () {};
    var pollIntervalMs = opts.pollIntervalMs || 1000;
    var maxReconnectMs = opts.maxReconnectMs || 30000;

    var st = { mode: "idle", es: null, pollTimer: null, stopped: false, attempts: 0 };

    function dispatch(data) {
      try { onState(data); }
      catch (e) { if (global.console && global.console.error) global.console.error("AuraStream onState:", e); }
    }

    function startPolling() {
      if (st.pollTimer || st.stopped) return;
      st.mode = "polling";
      (function tick() {
        if (st.stopped) return;
        fetch(pollUrl, { cache: "no-store" })
          .then(function (r) { return r.ok ? r.json() : null; })
          .then(function (j) { if (j) dispatch(j); })
          .catch(function () {})
          .finally(function () {
            if (!st.stopped) st.pollTimer = setTimeout(tick, pollIntervalMs);
          });
      })();
    }

    function stopPolling() {
      if (st.pollTimer) { clearTimeout(st.pollTimer); st.pollTimer = null; }
    }

    function connect() {
      if (st.stopped) return;
      if (typeof global.EventSource !== "function" || !url) { startPolling(); return; }
      var es;
      try { es = new global.EventSource(url); }
      catch (e) { startPolling(); return; }
      st.es = es;
      st.mode = "connecting";
      es.onopen = function () { st.mode = "stream"; st.attempts = 0; stopPolling(); };
      es.onmessage = function (ev) {
        try { dispatch(JSON.parse(ev.data)); } catch (e) {}
      };
      es.onerror = function () {
        try { es.close(); } catch (e) {}
        st.es = null;
        st.mode = "reconnecting";
        startPolling();
        if (st.stopped) return;
        var delay = Math.min(maxReconnectMs, 1000 * Math.pow(2, st.attempts++));
        setTimeout(connect, delay);
      };
    }

    connect();

    return {
      stop: function () {
        st.stopped = true;
        stopPolling();
        if (st.es) { try { st.es.close(); } catch (e) {} st.es = null; }
      },
      get mode() { return st.mode; }
    };
  };

  global.AuraStream = AuraStream;
})(window);

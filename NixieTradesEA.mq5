//+------------------------------------------------------------------+
//|  NixTradesEA.mq5                                                 |
//|  Nixie Trades - Client-Side Execution Expert Advisor             |
//|  Version: 2.1.0                                                  |
//|  Role: Nix Trades Quantitative Development Team                  |
//|                                                                   |
//|  HOW THIS WORKS                                                   |
//|  This EA runs silently on YOUR OWN MetaTrader 5 desktop app.     |
//|  Every 5 seconds it checks the Nixie Trades server for new       |
//|  trade signals assigned to your account. When a signal arrives   |
//|  it places the trade automatically on your MT5, then sends the   |
//|  result back so you receive a Telegram confirmation.             |
//|                                                                   |
//|  WHAT IS THE "SERVER URL"?                                        |
//|  The Server URL is the address of the Nixie Trades server.       |
//|  YOU do not host or set this up yourself.                        |
//|  Nixie Trades provides this URL to you when you subscribe.       |
//|  It looks like: http://123.45.67.89:8000                         |
//|  Just copy it exactly as given. Do not change anything in it.    |
//|                                                                   |
//|  IMPORTANT: KEEP MT5 OPEN                                        |
//|  Auto-trading only works while MetaTrader 5 is running.          |
//|  If your computer sleeps or MT5 is closed, trades stop.          |
//|  For 24/7 auto-trading, use a Windows VPS.                       |
//|  Type /vps_guide in the Nixie Trades Telegram bot for help.      |
//|                                                                   |
//|  NO EMOJIS - Enterprise-grade production code only               |
//+------------------------------------------------------------------+

#property copyright "Nixie Trades"
#property version   "2.10"
#property strict

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>

// ==================== CONSTANTS ====================

#define EA_VERSION    "2.1.0"
#define MAGIC_NUMBER  234567

// ==================== INPUTS ====================

input long   InpTelegramId      = 0;     // Your Telegram ID - find it by messaging @userinfobot on Telegram
input string InpWorkerUrl       = "";    // Nixie Trades Server URL - provided by Nixie Trades after signup
input string InpApiKey          = "";    // Nixie Trades API Key - provided by Nixie Trades after signup
input int    InpPollIntervalSec = 5;     // Signal check frequency in seconds (default 5)
input double InpDefaultRiskPct  = 1.0;  // Risk per trade as % of balance (default 1.0)
input int    InpSlippagePips    = 3;     // Maximum slippage in pips
input bool   InpVerboseLog      = false; // Show detailed logs (only enable for troubleshooting)

// ==================== GLOBALS ====================

CTrade        Trade;
CPositionInfo PositionInfo;

datetime  g_LastPollTime        = 0;
datetime  g_LastAccountSyncTime = 0;

// ==================== INITIALISATION ====================

int OnInit()
{
    if (InpTelegramId == 0)
    {
        Alert(
            "Nixie Trades EA: Your Telegram ID is missing.\n\n"
            "How to find it:\n"
            "1. Open Telegram on your phone or PC\n"
            "2. Search for @userinfobot\n"
            "3. Tap Start\n"
            "4. The bot replies with your ID number\n"
            "5. Copy that number and paste it into the EA settings"
        );
        return INIT_PARAMETERS_INCORRECT;
    }

    if (StringLen(InpWorkerUrl) < 7)
    {
        Alert(
            "Nixie Trades EA: The Server URL is missing.\n\n"
            "The Server URL is provided by Nixie Trades.\n"
            "Check your Telegram bot for the setup message, or contact @Nixiestone"
        );
        return INIT_PARAMETERS_INCORRECT;
    }

    if (StringLen(InpApiKey) < 8)
    {
        Alert(
            "Nixie Trades EA: The API Key is missing.\n\n"
            "The API Key is provided by Nixie Trades.\n"
            "Check your Telegram bot for the setup message, or contact @Nixiestone"
        );
        return INIT_PARAMETERS_INCORRECT;
    }

    Trade.SetExpertMagicNumber(MAGIC_NUMBER);
    Trade.SetDeviationInPoints(InpSlippagePips * 10);
    Trade.SetTypeFillingBySymbol(_Symbol);
    Trade.LogLevel(LOG_LEVEL_NO);

    Print("Nixie Trades EA v", EA_VERSION, " started. TelegramId=", InpTelegramId);
    Print(
        "NOTE: Keep MetaTrader 5 open at all times for auto-trading. "
        "If your computer sleeps, trading stops. "
        "For 24/7 trading without leaving your PC on, use a Windows VPS. "
        "Type /vps_guide in the Nixie Trades bot for a free beginner setup guide."
    );

    SyncAccountSnapshot();
    return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
    Print("Nixie Trades EA stopped. Reason=", reason);
}

// ==================== MAIN LOOP ====================

void OnTick()
{
    datetime now = TimeCurrent();

    if ((now - g_LastPollTime) >= InpPollIntervalSec)
    {
        g_LastPollTime = now;
        PollAndExecuteSignals();
        PollAndExecuteInstructions();
    }

    if ((now - g_LastAccountSyncTime) >= 60)
    {
        g_LastAccountSyncTime = now;
        SyncAccountSnapshot();
        SyncOpenPositions();
    }
}

// ==================== SIGNAL POLLING ====================

void PollAndExecuteSignals()
{
    string body     = "{\"telegram_id\":" + IntegerToString(InpTelegramId) + "}";
    string response = HttpPost(InpWorkerUrl + "/ea/pending_signals", body);

    if (StringLen(response) == 0) return;
    if (StringFind(response, "\"signals\":[]") >= 0) return;
    if (StringFind(response, "\"success\":false") >= 0)
    {
        if (InpVerboseLog) Print("Nixie Trades EA: signal poll error=", response);
        return;
    }

    int cursor = 0;
    int maxLen = StringLen(response);

    while (cursor < maxLen)
    {
        int keyPos = StringFind(response, "\"signal_id\":", cursor);
        if (keyPos < 0) break;

        int blkStart = FindBlockStart(response, keyPos);
        int blkEnd   = FindBlockEnd(response, blkStart);
        if (blkStart < 0 || blkEnd < 0) break;

        string signalJson = StringSubstr(response, blkStart, blkEnd - blkStart + 1);
        ExecuteSignal(signalJson);

        cursor = blkEnd + 1;
    }
}

void ExecuteSignal(const string signalJson)
{
    string signalId  = ExtractJsonString(signalJson, "signal_id");
    string symbol    = ExtractJsonString(signalJson, "symbol");
    string direction = ExtractJsonString(signalJson, "direction");
    string orderType = ExtractJsonString(signalJson, "order_type");
    double entry     = ExtractJsonDouble(signalJson, "entry");
    double sl        = ExtractJsonDouble(signalJson, "sl");
    double tp1       = ExtractJsonDouble(signalJson, "tp1");
    double riskPct   = ExtractJsonDouble(signalJson, "risk_percent");
    string comment   = ExtractJsonString(signalJson, "comment");

    if (StringLen(signalId) == 0 || StringLen(symbol) == 0) return;

    if (riskPct <= 0.0) riskPct = InpDefaultRiskPct;
    if (StringLen(comment) == 0) comment = "NT";

    Print("Nixie Trades EA: Signal received ", signalId,
          " | ", symbol, " ", direction, " Entry=", entry, " SL=", sl, " TP=", tp1);

    string brokerSym = NormaliseSymbol(symbol);
    if (StringLen(brokerSym) == 0)
    {
        ReportSignalResult(signalId, false, 0, 0.0, 0.0,
                           "Symbol not on this broker: " + symbol);
        return;
    }

    double pt     = SymbolInfoDouble(brokerSym, SYMBOL_POINT);
    double slPips = (pt > 0.0) ? MathAbs(entry - sl) / pt / 10.0 : 10.0;
    double lots   = CalculateLotSize(brokerSym, riskPct, slPips,
                                     AccountInfoDouble(ACCOUNT_BALANCE));

    if (lots <= 0.0)
    {
        ReportSignalResult(signalId, false, 0, 0.0, 0.0, "Lot size calculation returned zero.");
        return;
    }

    bool   ok    = false;
    long   tic   = 0;
    double aEnt  = 0.0;
    string errMsg = "";

    if      (orderType == "MARKET") ok = PlaceMarketOrder(brokerSym, direction, sl, tp1, lots, comment, tic, aEnt, errMsg);
    else if (orderType == "LIMIT")  ok = PlaceLimitOrder (brokerSym, direction, entry, sl, tp1, lots, comment, tic, aEnt, errMsg);
    else if (orderType == "STOP")   ok = PlaceStopOrder  (brokerSym, direction, entry, sl, tp1, lots, comment, tic, aEnt, errMsg);
    else                             errMsg = "Unknown order type: " + orderType;

    ReportSignalResult(signalId, ok, tic, aEnt, lots, errMsg);

    if (ok)   Print("Nixie Trades EA: Trade placed ticket=", tic, " lots=", lots, " entry=", aEnt);
    else      Print("Nixie Trades EA: Trade failed reason=", errMsg);
}

// ==================== INSTRUCTION POLLING ====================

void PollAndExecuteInstructions()
{
    string body     = "{\"telegram_id\":" + IntegerToString(InpTelegramId) + "}";
    string response = HttpPost(InpWorkerUrl + "/ea/pending_instructions", body);

    if (StringLen(response) == 0) return;
    if (StringFind(response, "\"instructions\":[]") >= 0) return;

    int cursor = 0;
    int maxLen = StringLen(response);

    while (cursor < maxLen)
    {
        int keyPos = StringFind(response, "\"instruction_id\":", cursor);
        if (keyPos < 0) break;

        int blkStart = FindBlockStart(response, keyPos);
        int blkEnd   = FindBlockEnd(response, blkStart);
        if (blkStart < 0 || blkEnd < 0) break;

        string instrJson = StringSubstr(response, blkStart, blkEnd - blkStart + 1);
        ExecuteInstruction(instrJson);

        cursor = blkEnd + 1;
    }
}

void ExecuteInstruction(const string instrJson)
{
    string iId    = ExtractJsonString(instrJson, "instruction_id");
    string action = ExtractJsonString(instrJson, "action");
    long   tic    = (long)ExtractJsonDouble(instrJson, "ticket");

    if (StringLen(iId) == 0 || StringLen(action) == 0 || tic == 0) return;

    bool   ok  = false;
    string err = "";

    if      (action == "close_partial") { double pct = ExtractJsonDouble(instrJson, "close_pct"); ok = ClosePartialPosition(tic, pct, err); }
    else if (action == "modify_sl")     { double nsl = ExtractJsonDouble(instrJson, "new_sl");    ok = ModifyPositionSl(tic, nsl, err);     }
    else                                 err = "Unknown action: " + action;

    ReportInstructionResult(iId, ok, err);
    Print("Nixie Trades EA: Instruction ", (ok ? "done" : "failed"), " id=", iId, " err=", err);
}

// ==================== ORDER PLACEMENT ====================

bool PlaceMarketOrder(
    const string sym, const string dir,
    const double sl, const double tp, const double lots,
    const string cmt,
    long &tic, double &aEnt, string &err
)
{
    bool isBuy = (dir == "BUY");
    double px  = isBuy ? SymbolInfoDouble(sym, SYMBOL_ASK) : SymbolInfoDouble(sym, SYMBOL_BID);
    bool ok    = isBuy ? Trade.Buy(lots, sym, px, sl, tp, cmt)
                       : Trade.Sell(lots, sym, px, sl, tp, cmt);
    if (!ok) { err = "Market order failed: " + Trade.ResultRetcodeDescription(); return false; }
    tic  = Trade.ResultOrder();
    aEnt = Trade.ResultPrice();
    return true;
}

bool PlaceLimitOrder(
    const string sym, const string dir,
    const double entry, const double sl, const double tp,
    const double lots, const string cmt,
    long &tic, double &aEnt, string &err
)
{
    ENUM_ORDER_TYPE t = (dir == "BUY") ? ORDER_TYPE_BUY_LIMIT : ORDER_TYPE_SELL_LIMIT;
    bool ok = Trade.OrderOpen(sym, t, lots, 0.0, entry, sl, tp,
                               ORDER_TIME_SPECIFIED, TimeCurrent() + 3600, cmt);
    if (!ok) { err = "Limit order failed: " + Trade.ResultRetcodeDescription(); return false; }
    tic  = Trade.ResultOrder();
    aEnt = entry;
    return true;
}

bool PlaceStopOrder(
    const string sym, const string dir,
    const double entry, const double sl, const double tp,
    const double lots, const string cmt,
    long &tic, double &aEnt, string &err
)
{
    ENUM_ORDER_TYPE t = (dir == "BUY") ? ORDER_TYPE_BUY_STOP : ORDER_TYPE_SELL_STOP;
    bool ok = Trade.OrderOpen(sym, t, lots, 0.0, entry, sl, tp,
                               ORDER_TIME_SPECIFIED, TimeCurrent() + 3600, cmt);
    if (!ok) { err = "Stop order failed: " + Trade.ResultRetcodeDescription(); return false; }
    tic  = Trade.ResultOrder();
    aEnt = entry;
    return true;
}

bool ClosePartialPosition(const long tic, const double pct, string &err)
{
    if (!PositionInfo.SelectByTicket(tic)) { err = "Position not found: " + IntegerToString(tic); return false; }
    double vol = NormalizeDouble(PositionInfo.Volume() * pct, 2);
    double mn  = SymbolInfoDouble(PositionInfo.Symbol(), SYMBOL_VOLUME_MIN);
    if (vol < mn) vol = mn;
    if (!Trade.PositionClosePartial(tic, vol)) { err = "Partial close failed: " + Trade.ResultRetcodeDescription(); return false; }
    return true;
}

bool ModifyPositionSl(const long tic, const double nsl, string &err)
{
    if (!PositionInfo.SelectByTicket(tic)) { err = "Position not found: " + IntegerToString(tic); return false; }
    if (!Trade.PositionModify(tic, nsl, PositionInfo.TakeProfit())) { err = "SL modify failed: " + Trade.ResultRetcodeDescription(); return false; }
    return true;
}

// ==================== LOT SIZE ====================

double CalculateLotSize(
    const string sym, const double riskPct,
    const double slPips, const double balance
)
{
    if (slPips <= 0.0 || balance <= 0.0) return SymbolInfoDouble(sym, SYMBOL_VOLUME_MIN);

    double tv  = SymbolInfoDouble(sym, SYMBOL_TRADE_TICK_VALUE);
    double ts  = SymbolInfoDouble(sym, SYMBOL_TRADE_TICK_SIZE);
    double pt  = SymbolInfoDouble(sym, SYMBOL_POINT);
    double vmn = SymbolInfoDouble(sym, SYMBOL_VOLUME_MIN);
    double vmx = SymbolInfoDouble(sym, SYMBOL_VOLUME_MAX);
    double vst = SymbolInfoDouble(sym, SYMBOL_VOLUME_STEP);

    if (ts <= 0.0 || pt <= 0.0) return vmn;

    double ppv = tv * (pt / ts) * 10.0;
    if (ppv <= 0.0) return vmn;

    double raw = (balance * riskPct / 100.0) / (slPips * ppv);
    if (vst > 0.0) raw = MathFloor(raw / vst) * vst;
    return NormalizeDouble(MathMax(vmn, MathMin(vmx, raw)), 2);
}

// ==================== SYMBOL NORMALISATION ====================

string NormaliseSymbol(const string raw)
{
    if (SymbolInfoInteger(raw, SYMBOL_EXIST) != 0 && SymbolSelect(raw, true)) return raw;

    string sfx[] = { ".pro",".raw",".m",".i",".a",".b",".c","_",".ecn",".std",".fix",".stp",".r" };
    for (int i = 0; i < ArraySize(sfx); i++)
    {
        string c = raw + sfx[i];
        if (SymbolInfoInteger(c, SYMBOL_EXIST) != 0 && SymbolSelect(c, true)) return c;
    }

    for (int i = 0; i < SymbolsTotal(false); i++)
    {
        string s = SymbolName(i, false);
        if (StringFind(s, raw) == 0) { SymbolSelect(s, true); return s; }
    }

    Print("Nixie Trades EA: Symbol not found on broker: ", raw);
    return "";
}

// ==================== REPORTING ====================

void ReportSignalResult(
    const string sid, const bool ok, const long tic,
    const double aEnt, const double lots, const string err
)
{
    string body =
        "{\"signal_id\":\"" + sid + "\","
        + "\"telegram_id\":" + IntegerToString(InpTelegramId) + ","
        + "\"status\":\"" + (ok ? "executed" : "failed") + "\","
        + "\"order_ticket\":" + IntegerToString(tic) + ","
        + "\"actual_entry\":" + DoubleToString(aEnt, 5) + ","
        + "\"actual_lots\":" + DoubleToString(lots, 2) + ","
        + "\"ea_error\":\"" + EscapeJsonString(err) + "\"}";
    HttpPost(InpWorkerUrl + "/ea/signal_result", body);
}

void ReportInstructionResult(const string iid, const bool ok, const string err)
{
    string body =
        "{\"instruction_id\":\"" + iid + "\","
        + "\"telegram_id\":" + IntegerToString(InpTelegramId) + ","
        + "\"status\":\"" + (ok ? "executed" : "failed") + "\","
        + "\"ea_error\":\"" + EscapeJsonString(err) + "\"}";
    HttpPost(InpWorkerUrl + "/ea/instruction_result", body);
}

void SyncAccountSnapshot()
{
    string body =
        "{\"telegram_id\":" + IntegerToString(InpTelegramId) + ","
        + "\"balance\":"    + DoubleToString(AccountInfoDouble(ACCOUNT_BALANCE), 2) + ","
        + "\"equity\":"     + DoubleToString(AccountInfoDouble(ACCOUNT_EQUITY), 2) + ","
        + "\"leverage\":"   + IntegerToString((int)AccountInfoInteger(ACCOUNT_LEVERAGE)) + ","
        + "\"currency\":\"" + AccountInfoString(ACCOUNT_CURRENCY) + "\","
        + "\"broker\":\""   + EscapeJsonString(AccountInfoString(ACCOUNT_COMPANY)) + "\","
        + "\"server\":\""   + EscapeJsonString(AccountInfoString(ACCOUNT_SERVER)) + "\"}";
    HttpPost(InpWorkerUrl + "/ea/account_sync", body);
    if (InpVerboseLog) Print("Nixie Trades EA: Account synced balance=", AccountInfoDouble(ACCOUNT_BALANCE));
}

void SyncOpenPositions()
{
    string arr = "[";
    bool   f   = true;

    for (int i = PositionsTotal() - 1; i >= 0; i--)
    {
        if (!PositionInfo.SelectByIndex(i)) continue;
        if (PositionInfo.Magic() != MAGIC_NUMBER) continue;
        if (!f) arr += ",";
        f = false;
        arr +=
            "{\"ticket\":"     + IntegerToString(PositionInfo.Ticket()) + ","
            + "\"symbol\":\""  + PositionInfo.Symbol() + "\","
            + "\"direction\":\"" + (PositionInfo.PositionType() == POSITION_TYPE_BUY ? "BUY" : "SELL") + "\","
            + "\"lots\":"      + DoubleToString(PositionInfo.Volume(), 2) + ","
            + "\"entry\":"     + DoubleToString(PositionInfo.PriceOpen(), 5) + ","
            + "\"sl\":"        + DoubleToString(PositionInfo.StopLoss(), 5) + ","
            + "\"tp\":"        + DoubleToString(PositionInfo.TakeProfit(), 5) + ","
            + "\"profit\":"    + DoubleToString(PositionInfo.Profit(), 2) + "}";
    }
    arr += "]";

    string body =
        "{\"telegram_id\":" + IntegerToString(InpTelegramId) + ","
        + "\"positions\":" + arr + "}";
    HttpPost(InpWorkerUrl + "/ea/positions_sync", body);
}

// ==================== HTTP ====================

string HttpPost(const string url, const string body)
{
    char   req[];
    char   res[];
    string resH;
    StringToCharArray(body, req, 0, StringLen(body));
    string hdr = "Content-Type: application/json\r\nX-API-Key: " + InpApiKey + "\r\n";
    int code   = WebRequest("POST", url, hdr, 10000, req, res, resH);
    if (code == -1)
    {
        int e = GetLastError();
        if (e == 4060)
            Print("Nixie Trades EA: URL not allowed. Tools > Options > Expert Advisors > Allow WebRequest > add: ", url);
        else if (InpVerboseLog)
            Print("Nixie Trades EA: HTTP error url=", url, " code=", e);
        return "";
    }
    return CharArrayToString(res, 0, ArraySize(res));
}

// ==================== JSON ====================

string ExtractJsonString(const string json, const string key)
{
    string s = "\"" + key + "\":\"";
    int p    = StringFind(json, s);
    if (p < 0) return "";
    p += StringLen(s);
    int e = StringFind(json, "\"", p);
    if (e < 0) return "";
    return StringSubstr(json, p, e - p);
}

double ExtractJsonDouble(const string json, const string key)
{
    string s = "\"" + key + "\":";
    int p    = StringFind(json, s);
    if (p < 0) return 0.0;
    p += StringLen(s);
    if (StringSubstr(json, p, 4) == "null") return 0.0;
    string v = "";
    for (int i = p; i < MathMin(p + 32, StringLen(json)); i++)
    {
        ushort c = StringGetCharacter(json, i);
        if (c == ',' || c == '}' || c == ']') break;
        v += ShortToString(c);
    }
    return StringToDouble(v);
}

int FindBlockStart(const string json, const int from)
{
    for (int i = from; i >= 0; i--)
        if (StringGetCharacter(json, i) == '{') return i;
    return -1;
}

int FindBlockEnd(const string json, const int start)
{
    if (start < 0) return -1;
    int d = 0, n = StringLen(json);
    for (int i = start; i < n; i++)
    {
        ushort c = StringGetCharacter(json, i);
        if      (c == '{') d++;
        else if (c == '}') { d--; if (d == 0) return i; }
    }
    return -1;
}

string EscapeJsonString(string v)
{
    StringReplace(v, "\\", "\\\\");
    StringReplace(v, "\"", "\\\"");
    StringReplace(v, "\n", "\\n");
    StringReplace(v, "\r", "\\r");
    return v;
}
//+------------------------------------------------------------------+
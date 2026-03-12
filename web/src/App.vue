<template>
  <div class="app">
    <header class="hero">
      <div>
        <h1>StockQuantBot</h1>
        <p>短线市场监控 · 规则列表 + 模型Top</p>
      </div>
      <div class="status">
        <div class="chip" :class="marketOpen ? 'open' : 'closed'">
          {{ marketOpen ? '交易中' : '已休市' }}
        </div>
        <div class="chip muted">{{ serverTimeLabel }}</div>
      </div>
    </header>

    <section class="view-tabs">
      <button class="tab-btn" :class="{ active: viewPage === 'market' }" @click="viewPage = 'market'">
        看盘面板
      </button>
      <button class="tab-btn" :class="{ active: viewPage === 'review' }" @click="viewPage = 'review'">
        复盘助手
      </button>
    </section>

    <section class="controls" v-if="viewPage === 'market'">
      <div class="control">
        <label>模式</label>
        <select v-model="mode" @change="refreshNow">
          <option value="all">全市场扫描</option>
          <option value="watchlist">自选列表</option>
        </select>
      </div>
      <div class="control">
        <label>行情源</label>
        <select v-model="provider" @change="refreshNow">
          <option value="biying">必盈</option>
          <option value="tencent">腾讯</option>
          <option value="auto">自动</option>
        </select>
      </div>
      <div class="control model-switch">
        <label>模型</label>
        <div class="model-buttons">
          <button
            v-for="opt in modelOptions"
            :key="opt.key || '__default__'"
            class="ghost small"
            :class="{ active: modelKey === opt.key }"
            @click="setModelKey(opt.key)"
          >
            {{ opt.label }}
          </button>
        </div>
      </div>
      <div class="control">
        <label>模型独立</label>
        <input type="checkbox" v-model="modelIndependent" @change="refreshModelTop" />
      </div>
      <div class="control">
        <label>模型板块</label>
        <input
          type="text"
          v-model.trim="modelSector"
          placeholder="如 半导体/消费"
          @keydown.enter.prevent="refreshModelTop"
        />
      </div>
      <div class="control">
        <label>只看 BUY</label>
        <input type="checkbox" v-model="onlyBuy" @change="refreshNow" />
      </div>
      <div class="control actions">
        <button @click="refreshNow">立即刷新</button>
      </div>
    </section>

    <section class="grid" v-if="viewPage === 'market'">
      <div class="col">
        <div class="panel">
          <div class="panel-header">
            <h2>规则列表</h2>
            <span class="meta">{{ rows.length }} 条</span>
            <span class="meta" v-if="perfLabel">{{ perfLabel }}</span>
          </div>
          <div class="table-wrap" @contextmenu.prevent>
            <table>
              <colgroup>
                <col class="col-symbol" />
                <col class="col-name" />
                <col class="col-action" />
                <col class="col-sector" />
                <col class="col-score" />
                <col class="col-score" />
                <col class="col-score" />
                <col class="col-pct" />
                <col class="col-reason" />
                <col class="col-ai" />
              </colgroup>
              <thead>
                <tr>
                  <th class="sortable col-symbol" @click="setSort('symbol')">代码 <span class="arrow">{{ sortArrow('symbol') }}</span></th>
                  <th class="sortable col-name" @click="setSort('name')">名称 <span class="arrow">{{ sortArrow('name') }}</span></th>
                  <th class="sortable col-action" @click="setSort('action')">动作 <span class="arrow">{{ sortArrow('action') }}</span></th>
                  <th class="sortable col-sector" @click="setSort('sector')">板块 <span class="arrow">{{ sortArrow('sector') }}</span></th>
                  <th class="sortable col-score" @click="setSort('combo')">综合 <span class="arrow">{{ sortArrow('combo') }}</span></th>
                  <th class="sortable col-score" @click="setSort('score')">评分 <span class="arrow">{{ sortArrow('score') }}</span></th>
                  <th class="sortable col-score" @click="setSort('model_score')">模型% <span class="arrow">{{ sortArrow('model_score') }}</span></th>
                  <th class="sortable col-pct" @click="setSort('pct_chg')">涨跌幅 <span class="arrow">{{ sortArrow('pct_chg') }}</span></th>
                  <th class="sortable col-reason" @click="setSort('reason')">原因 <span class="arrow">{{ sortArrow('reason') }}</span></th>
                  <th class="col-ai">AI解释</th>
                </tr>
              </thead>
              <tbody>
                <tr
                  v-for="row in sortedRows"
                  :key="row.symbol"
                  @contextmenu.prevent="openContextMenu($event, row.symbol)"
                >
                  <td class="mono col-symbol">{{ row.symbol }}</td>
                  <td class="col-name">{{ row.name }}</td>
                  <td class="col-action"><span :class="['badge', row.action?.toLowerCase()]">{{ row.action }}</span></td>
                  <td class="col-sector">{{ row.sector || '--' }}</td>
                  <td class="col-score">{{ fmt(comboScore(row)) }}</td>
                  <td class="col-score">{{ fmt(row.score) }}</td>
                  <td class="col-score">{{ fmtPct(row.model_score) }}</td>
                  <td class="col-pct" :class="row.pct_chg >= 0 ? 'up' : 'down'">{{ fmt(row.pct_chg) }}</td>
                  <td class="reason col-reason">{{ row.reason }}</td>
                  <td class="ai col-ai">
                    <button class="ghost small" @click="openExplain(row)">
                      {{ explainLabel(row.symbol) }}
                    </button>
                  </td>
                </tr>
              </tbody>
            </table>
            <div v-if="loading" class="overlay">加载中...</div>
            <div v-if="!loading && rows.length === 0" class="overlay empty">暂无结果</div>
          </div>
        </div>

        <section class="panel news">
          <div class="panel-header">
            <h2>新闻</h2>
            <span class="meta">{{ newsItems.length }} 条</span>
          </div>
          <div class="summary-card">
            <div v-if="newsSummaryLoading" class="muted">情绪摘要加载中...</div>
            <div v-else-if="newsSummaryError" class="muted">{{ newsSummaryError }}</div>
            <div v-else-if="!newsSummary" class="muted">暂无情绪摘要</div>
            <div v-else class="summary-grid">
              <div>
                <div class="summary-label">市场情绪</div>
                <div class="summary-value">
                  {{ newsSummary.market_sentiment || 'neutral' }}
                  <span class="summary-score">{{ newsSummary.score ?? 50 }}</span>
                </div>
              </div>
              <div>
                <div class="summary-label">风险等级</div>
                <div class="summary-value">{{ newsSummary.risk_level || 'medium' }}</div>
              </div>
              <div>
                <div class="summary-label">热点板块</div>
                <div class="summary-value">
                  {{ (newsSummary.hot_sectors || []).join('、') || '暂无' }}
                </div>
              </div>
              <div class="summary-full">
                <div class="summary-label">风格建议</div>
                <div class="summary-text">{{ newsSummary.suggested_style || '暂无' }}</div>
              </div>
              <div class="summary-full">
                <div class="summary-label">点评</div>
                <div class="summary-text">{{ newsSummary.comment || '暂无' }}</div>
              </div>
            </div>
          </div>
          <div class="news-controls">
            <select v-model="newsSymbol" @change="fetchNews">
              <option value="">市场</option>
              <option v-for="item in watchlist" :key="item.symbol" :value="item.symbol">
                {{ item.name || item.symbol }}
              </option>
            </select>
            <button class="ghost" @click="fetchNews">刷新</button>
          </div>
          <div class="news-list">
            <div v-if="newsLoading" class="muted">加载中...</div>
            <div v-else-if="newsError" class="muted">{{ newsError }}</div>
            <div v-else-if="newsItems.length === 0" class="muted">暂无新闻</div>
            <a
              v-for="item in newsItems"
              :key="item.url || item.title"
              class="news-item"
              :href="item.url || '#'"
              target="_blank"
              rel="noreferrer"
            >
              <div class="news-title">{{ item.title }}</div>
              <div class="news-meta">
                <span>{{ item.name || (item.symbol ? item.symbol : '市场') }}</span>
                <span class="sentiment" :class="sentimentClass(item.sentiment)">{{ item.sentiment || '中性' }}</span>
                <span>{{ item.source || '资讯' }}</span>
                <span>{{ item.time || '' }}</span>
              </div>
            </a>
          </div>
        </section>

        <section class="panel watchlist">
          <div class="panel-header">
            <h2>自选</h2>
            <span class="meta">{{ watchlist.length }} 只</span>
          </div>
          <div v-if="isWatchlist" class="watchlist-add">
            <input v-model.trim="watchlistInput" placeholder="输入代码，如 000001 / 600000" />
            <button @click="addFromInput">添加</button>
          </div>
          <div class="chips">
            <span v-for="item in watchlist" :key="item.symbol" class="chip" @click="removeFromWatchlist(item.symbol)">
              {{ item.symbol }} ×
            </span>
            <span v-if="watchlist.length === 0" class="muted">暂无自选</span>
          </div>
        </section>
      </div>

      <aside class="col side">
        <div class="panel">
          <div class="panel-header">
            <h2>模型 Top</h2>
            <span class="meta">{{ modelTop.length }} 条</span>
            <span class="meta" v-if="modelLoading">刷新中...</span>
          </div>
          <div class="note" v-if="!isWatchlist">右键任意行添加到自选</div>
          <div v-if="modelLoading" class="muted">模型候选刷新中...</div>
          <div v-else-if="modelWarmupPending" class="muted">模型榜单预热中，稍后自动补齐...</div>
          <div v-else-if="modelTop.length === 0" class="muted">暂无模型结果</div>
          <ul v-else class="list">
            <li v-for="row in modelTop" :key="row.symbol" @contextmenu.prevent="openContextMenu($event, row.symbol)">
              <div>
                <div class="title">
                  <span class="mono">{{ row.symbol }}</span>
                  <span>{{ row.name }}</span>
                </div>
                <div class="sub">
                  <span>模型 {{ fmtPct(row.model_score) }}</span>
                  <span v-if="row.sector">{{ row.sector }}</span>
                  <span :class="row.pct_chg >= 0 ? 'up' : 'down'">{{ fmt(row.pct_chg) }}%</span>
                </div>
              </div>
              <button v-if="!isWatchlist" class="ghost" @click="addToWatchlist(row.symbol)">+自选</button>
            </li>
          </ul>
        </div>
      </aside>
    </section>

    <section v-else class="review-page">
      <div class="review-layout">
        <section class="panel review-chat-panel">
          <div class="panel-header">
            <h2>复盘聊天</h2>
            <span class="meta">{{ reviewDate }}</span>
          </div>
          <div class="review-date-row">
            <label>复盘日期</label>
            <input type="date" v-model="reviewDate" @change="loadReviewJournal" />
          </div>
          <div class="chat-list">
            <div v-if="reviewMessages.length === 0" class="muted">先说今天做了什么，我会自动抽取操作并追问缺失信息。</div>
            <div
              v-for="msg in reviewMessages"
              :key="msg.id"
              class="chat-item"
              :class="msg.role === 'assistant' ? 'assistant' : 'user'"
            >
              <div class="chat-role">{{ msg.role === 'assistant' ? '助手' : '我' }}</div>
              <div class="chat-text">{{ msg.text }}</div>
            </div>
          </div>
          <textarea
            v-model="chatInput"
            class="chat-input"
            rows="3"
            placeholder="输入复盘内容，Enter发送，Shift+Enter换行（例：今天 600000 9.72 买入 2000股，+1.8% 止盈）"
            @keydown="onChatInputKeydown"
          ></textarea>
          <div class="muted">{{ chatSending ? '发送中...' : '聊天会自动抽取操作、关联行情并更新右侧笔记卡。' }}</div>
          <div class="review-actions single-action">
            <button class="ghost primary" @click="generateReview" :disabled="reviewLoading">
              {{ reviewLoading ? '生成中...' : '生成今日复盘' }}
            </button>
          </div>
          <div v-if="reviewError" class="muted">{{ reviewError }}</div>
          <div v-else-if="reviewText" class="review-text">{{ reviewText }}</div>
        </section>

        <section class="panel review-card-panel">
          <div class="panel-header">
            <h2>今日笔记卡</h2>
            <span class="meta" v-if="reviewSource">来源: {{ reviewSource }}</span>
          </div>
          <div class="card-block">
            <div class="card-title">操作清单（自动提取）</div>
            <div v-if="cardOperations.length === 0" class="muted">聊天中提到的交易会自动出现在这里</div>
            <div v-for="op in cardOperations" :key="op.id" class="card-op-item">
              <span class="mono">{{ op.symbol || '--' }}</span>
              <span>{{ op.side || '--' }}</span>
              <span>{{ fmt(op.price) }}</span>
              <span :class="Number(op.result_pct) >= 0 ? 'up' : 'down'">{{ op.result_pct == null ? '--' : fmt(op.result_pct) + '%' }}</span>
              <span class="muted">MA20 {{ fmt(op.ma20) }}</span>
              <span class="muted">波动 {{ fmt(op.atr_pct) }}%</span>
            </div>
          </div>
          <div class="card-block">
            <div class="card-title">今日结论</div>
            <div class="card-sub">做对</div>
            <div v-for="(x, idx) in cardRightPoints" :key="'r' + idx" class="card-line">{{ x }}</div>
            <div class="card-sub">待改进</div>
            <div v-for="(x, idx) in cardWrongPoints" :key="'w' + idx" class="card-line">{{ x }}</div>
            <div class="card-tag" v-if="reviewCard?.error_tag">错误标签：{{ reviewCard.error_tag }}</div>
            <div class="card-tag" v-if="monthlyTopError">本月高频错误：{{ monthlyTopError }}</div>
          </div>
          <div class="card-block">
            <div class="card-title">明日计划（3条触发规则）</div>
            <div v-for="(x, idx) in cardPlans" :key="'p' + idx" class="card-line">{{ idx + 1 }}. {{ x }}</div>
            <div class="muted" v-if="monitorConfigPath">已写入监控配置：{{ monitorConfigPath }}</div>
          </div>
        </section>
      </div>
    </section>

    <div v-if="contextMenu.show && !isWatchlist" class="context" :style="contextStyle" @click="contextMenu.show=false">
      <button @click.stop="addToWatchlist(contextMenu.symbol)">添加到自选</button>
    </div>

    <div v-if="explainOpen" class="modal" @click.self="explainOpen = false">
      <div class="modal-card">
        <div class="modal-header">
          <div>
            <div class="modal-title">{{ explainTitle }}</div>
            <div class="muted" v-if="explainSource">来源: {{ explainSource }}</div>
          </div>
          <button class="ghost" @click="explainOpen = false">关闭</button>
        </div>
        <div class="modal-body">
          <div class="level-grid" v-if="explainLevels && (explainLevels.entry || explainLevels.stop || explainLevels.target)">
            <div class="level">
              <div class="label">参考入场</div>
              <div class="value">{{ fmt(explainLevels.entry) }}</div>
            </div>
            <div class="level">
              <div class="label">参考止损</div>
              <div class="value">{{ fmt(explainLevels.stop) }}</div>
            </div>
            <div class="level">
              <div class="label">参考目标</div>
              <div class="value">{{ fmt(explainLevels.target) }}</div>
            </div>
          </div>
          <div v-else class="muted">暂无参考价位</div>
          <div v-if="explainLoading" class="muted">生成中...</div>
          <div v-else>{{ explainText }}</div>
          <div class="footnote">仅用于信号解释，不构成投资建议。</div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, computed, watch } from 'vue'

const rows = ref([])
const modelTop = ref([])
const notes = ref([])
const stats = ref({})
const loading = ref(false)
const modelLoading = ref(false)
const mode = ref('all')
const provider = ref('tencent')
const modelOptions = ref([{ key: '', label: 'default', path: '' }])
const modelKey = ref('')
const modelIndependent = ref(false)
const modelSector = ref('')
const onlyBuy = ref(false)
const topN = ref(20)
const marketOpen = ref(false)
const serverTime = ref('')
const watchlist = ref([])
const watchlistInput = ref('')
const newsItems = ref([])
const newsLoading = ref(false)
const newsError = ref('')
const newsSymbol = ref('')
const newsSummary = ref(null)
const newsSummaryLoading = ref(false)
const newsSummaryError = ref('')
const sortKey = ref('combo')
const sortDir = ref('desc')
const comboRuleWeight = 0.6
const comboModelWeight = 0.4
const explainOpen = ref(false)
const explainTitle = ref('')
const explainText = ref('')
const explainSource = ref('')
const explainLoading = ref(false)
const explainCache = ref({})
const explainLevels = ref(null)
const reviewText = ref('')
const reviewSource = ref('')
const reviewLoading = ref(false)
const reviewError = ref('')
const viewPage = ref('market')
const reviewDate = ref('')
const reviewMessages = ref([])
const reviewOperations = ref([])
const reviewCard = ref({})
const reviewErrorStats = ref([])
const chatInput = ref('')
const chatSending = ref(false)
const monitorConfigPath = ref('')

const contextMenu = ref({ show: false, x: 0, y: 0, symbol: '' })
const isWatchlist = computed(() => mode.value === 'watchlist')
const isReviewPage = computed(() => viewPage.value === 'review')
const contextStyle = computed(() => ({ left: contextMenu.value.x + 'px', top: contextMenu.value.y + 'px' }))
let refreshTimer = null
let refreshAbortController = null
let refreshRequestSeq = 0
let modelAbortController = null
let modelRequestSeq = 0
let modelWarmupRetryTimer = null
const apiOrigin = (() => {
  const envBase = String(import.meta.env.VITE_API_BASE || '').trim().replace(/\/+$/, '')
  if (envBase) return envBase
  if (import.meta.env.DEV) return 'http://127.0.0.1:8000'
  if (typeof window !== 'undefined' && window.location?.origin) return window.location.origin
  return ''
})()

function apiEndpoint(path) {
  const normalized = path.startsWith('/') ? path : `/${path}`
  return apiOrigin ? `${apiOrigin}${normalized}` : normalized
}

function apiUrl(path) {
  return new URL(apiEndpoint(path), window.location.origin)
}

const serverTimeLabel = computed(() => {
  if (!serverTime.value) return '--'
  return serverTime.value.replace('T', ' ').slice(0, 19)
})

const perfLabel = computed(() => {
  const s = stats.value || {}
  const total = Number(s.time_total_ms)
  if (!total || Number.isNaN(total)) return ''
  const spot = Number(s.time_spot_ms) || 0
  const factors = Number(s.time_factors_ms) || 0
  const model = Number(s.time_model_ms) || 0
  const score = Number(s.time_score_ms) || 0
  return `耗时 ${Math.round(total)}ms (行情${Math.round(spot)} 因子${Math.round(factors)} 模型${Math.round(model)} 评分${Math.round(score)})`
})

const modelWarmupPending = computed(() => {
  if (modelLoading.value || modelTop.value.length > 0) return false
  return (notes.value || []).some((note) => {
    const text = String(note || '')
    return text.includes('模型榜单首次生成中') || text.includes('后台刷新中') || text.includes('后台刷新已触发')
  })
})

const cardOperations = computed(() => {
  const items = reviewCard.value?.operations
  return Array.isArray(items) ? items : []
})

const cardRightPoints = computed(() => {
  const items = reviewCard.value?.right_points
  return Array.isArray(items) ? items : []
})

const cardWrongPoints = computed(() => {
  const items = reviewCard.value?.wrong_points
  return Array.isArray(items) ? items : []
})

const cardPlans = computed(() => {
  const items = reviewCard.value?.tomorrow_plan
  return Array.isArray(items) ? items : []
})

const monthlyTopError = computed(() => {
  const items = reviewErrorStats.value || []
  if (!Array.isArray(items) || items.length === 0) return ''
  return items.map((x) => `${x.tag}(${x.count})`).join('、')
})

function normalizeSectorName(val) {
  return String(val || '')
    .replace(/\s+/g, '')
    .replace(/[、，,]/g, '')
    .replace(/板块|概念股|概念|主题|产业/g, '')
    .trim()
}

const hotSectorSet = computed(() => {
  const list = newsSummary.value?.hot_sectors || []
  return new Set(list.map((s) => normalizeSectorName(s)).filter(Boolean))
})

function isHotSector(row) {
  if (row?.hot_sector) return true
  const sector = normalizeSectorName(row?.sector || '')
  if (!sector) return false
  if (hotSectorSet.value.has(sector)) return true
  for (const hot of hotSectorSet.value) {
    if (!hot) continue
    if (sector.includes(hot) || hot.includes(sector)) return true
  }
  return false
}

const sortedRows = computed(() => {
  const data = rows.value.slice().map((r) => ({
    ...r,
    _combo: comboScore(r)
  }))
  const key = sortKey.value
  if (!key) return data
  const dir = sortDir.value === 'asc' ? 1 : -1
  return data.sort((a, b) => {
    const av = key === 'combo' ? a?._combo : a?.[key]
    const bv = key === 'combo' ? b?._combo : b?.[key]
  const ha = isHotSector(a) ? 1 : 0
  const hb = isHotSector(b) ? 1 : 0
  if (key !== 'hot_sector' && ha !== hb) return hb - ha
  if (key === 'hot_sector') {
    return (ha - hb) * dir
  }
  if (key === 'symbol' || key === 'name' || key === 'action' || key === 'reason' || key === 'sector') {
      const sa = String(av ?? '')
      const sb = String(bv ?? '')
      return sa.localeCompare(sb, 'zh') * dir
    }
    const an = Number(av)
    const bn = Number(bv)
    if (Number.isNaN(an) && Number.isNaN(bn)) return 0
    if (Number.isNaN(an)) return 1
    if (Number.isNaN(bn)) return -1
    return (an - bn) * dir
  })
})

function fmt(val) {
  if (val === null || val === undefined || Number.isNaN(val)) return '--'
  return Number(val).toFixed(2)
}

function fmtPct(val) {
  if (val === null || val === undefined || Number.isNaN(val)) return '--'
  return (Number(val) * 100).toFixed(2)
}

function comboScore(row) {
  const rule = Number(row?.score)
  const model = Number(row?.model_score)
  const ruleScore = Number.isNaN(rule) ? 0 : rule
  const modelScore = Number.isNaN(model) ? 0 : model * 100
  return ruleScore * comboRuleWeight + modelScore * comboModelWeight
}

function openContextMenu(evt, symbol) {
  if (isWatchlist.value) return
  contextMenu.value = { show: true, x: evt.clientX, y: evt.clientY, symbol }
}

function setSort(key) {
  if (sortKey.value === key) {
    sortDir.value = sortDir.value === 'asc' ? 'desc' : 'asc'
  } else {
    sortKey.value = key
    sortDir.value = 'desc'
  }
}

function sortArrow(key) {
  if (sortKey.value !== key) return ''
  return sortDir.value === 'asc' ? '▲' : '▼'
}

function sentimentClass(label) {
  if (label === '正面') return 'pos'
  if (label === '负面') return 'neg'
  return 'neu'
}

function explainLabel(symbol) {
  if (!symbol) return 'AI解释'
  return explainCache.value[symbol] ? '查看' : 'AI解释'
}

async function fetchWatchlist() {
  const res = await fetch(apiEndpoint('/api/watchlist?include_name=true'))
  const data = await res.json()
  watchlist.value = data.items || (data.symbols || []).map((s) => ({ symbol: s, name: '' }))
}

async function fetchModelOptions() {
  try {
    const res = await fetch(apiEndpoint('/api/model-options'))
    const data = await res.json()
    const items = Array.isArray(data?.items) ? data.items : []
    modelOptions.value = items.length ? items : [{ key: '', label: 'default', path: '' }]
  } catch (e) {
    modelOptions.value = [{ key: '', label: 'default', path: '' }]
  }
}

function syncSelectedModelKey() {
  const exists = modelOptions.value.some((x) => String(x.key || '') === String(modelKey.value || ''))
  if (!exists) modelKey.value = ''
}

function setModelKey(key) {
  const k = String(key || '')
  if (modelKey.value === k) return
  modelKey.value = k
  refreshModelTop()
}

async function fetchNews() {
  newsLoading.value = true
  newsError.value = ''
  try {
    const url = apiUrl('/api/news')
    if (newsSymbol.value) url.searchParams.set('symbol', newsSymbol.value)
    url.searchParams.set('limit', '20')
    const res = await fetch(url)
    const data = await res.json()
    newsItems.value = data.items || []
    if (data.error) newsError.value = data.error
    await fetchNewsSummary()
  } catch (e) {
    newsError.value = '新闻获取失败'
  } finally {
    newsLoading.value = false
  }
}

async function fetchNewsSummary() {
  newsSummaryLoading.value = true
  newsSummaryError.value = ''
  try {
    const url = apiUrl('/api/news/summary')
    url.searchParams.set('limit', '30')
    const res = await fetch(url)
    const data = await res.json()
    newsSummary.value = data.summary || null
  } catch (e) {
    newsSummaryError.value = '情绪摘要获取失败'
  } finally {
    newsSummaryLoading.value = false
  }
}

async function addToWatchlist(symbol) {
  if (!symbol) return
  await fetch(apiEndpoint('/api/watchlist'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ symbols: [symbol] })
  })
  await fetchWatchlist()
  contextMenu.value.show = false
}

async function addFromInput() {
  const raw = watchlistInput.value
  if (!raw) return
  const symbols = raw
    .split(/[,\s]+/g)
    .map((s) => s.trim())
    .filter(Boolean)
  if (symbols.length === 0) return
  await fetch(apiEndpoint('/api/watchlist'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ symbols })
  })
  watchlistInput.value = ''
  await fetchWatchlist()
  if (isWatchlist.value) {
    await refreshNow()
  }
}

async function removeFromWatchlist(symbol) {
  if (!symbol) return
  await fetch(apiEndpoint('/api/watchlist'), {
    method: 'DELETE',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ symbols: [symbol] })
  })
  await fetchWatchlist()
}

async function refreshNow() {
  const requestSeq = ++refreshRequestSeq
  if (refreshAbortController) {
    refreshAbortController.abort()
  }
  if (modelAbortController) {
    modelAbortController.abort()
    modelAbortController = null
  }
  const controller = new AbortController()
  refreshAbortController = controller
  const fullModelSeq = ++modelRequestSeq
  const timeoutMs = modelIndependent.value ? 70000 : 50000
  let timedOut = false
  const timeoutId = setTimeout(() => {
    timedOut = true
    controller.abort()
  }, timeoutMs)
  loading.value = true
  modelLoading.value = true

  const applyMarketPayload = (data) => {
    rows.value = data.rows || []
    modelTop.value = data.model_top || []
    notes.value = data.notes || []
    stats.value = data.stats || {}
    marketOpen.value = data.market_open
    serverTime.value = data.server_time || ''
    scheduleModelWarmupRetry(data)
  }

  const fetchMarketPayload = async (signal, modelIndependentFlag) => {
    const url = apiUrl('/api/market')
    url.searchParams.set('mode', mode.value)
    url.searchParams.set('provider', provider.value)
    syncSelectedModelKey()
    if (modelKey.value) url.searchParams.set('model', modelKey.value)
    url.searchParams.set('model_independent', String(modelIndependentFlag))
    if (modelSector.value) url.searchParams.set('model_sector', modelSector.value)
    url.searchParams.set('top_n', String(topN.value))
    url.searchParams.set('only_buy', String(onlyBuy.value))
    url.searchParams.set('intraday', 'false')
    const res = await fetch(url, { signal })
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    return await res.json()
  }
  try {
    const data = await fetchMarketPayload(controller.signal, modelIndependent.value)
    if (requestSeq !== refreshRequestSeq) return
    applyMarketPayload(data)
  } catch (e) {
    if (requestSeq !== refreshRequestSeq) return
    const msg = (e && e.message) ? String(e.message) : 'request failed'
    const shouldFallback =
      modelIndependent.value && (
        (e && e.name === 'AbortError' && timedOut) ||
        msg.includes('HTTP 504')
      )
    if (shouldFallback) {
      const retryController = new AbortController()
      const retryTimeout = setTimeout(() => {
        retryController.abort()
      }, 50000)
      try {
        const data = await fetchMarketPayload(retryController.signal, false)
        if (requestSeq !== refreshRequestSeq) return
        modelIndependent.value = false
        applyMarketPayload(data)
        const fallbackNotes = Array.isArray(notes.value) ? notes.value.slice() : []
        fallbackNotes.unshift('模型独立超时，已自动回退到非独立模式')
        notes.value = fallbackNotes
        return
      } catch (retryErr) {
        if (requestSeq !== refreshRequestSeq) return
        const retryMsg = (retryErr && retryErr.message) ? retryErr.message : 'fallback failed'
        notes.value = [`请求失败: ${retryMsg}`]
        return
      } finally {
        clearTimeout(retryTimeout)
      }
    }
    if (e && e.name === 'AbortError') {
      if (timedOut) {
        const hint = modelIndependent.value
          ? '请求超时：全量模型候选过重，建议关闭“模型独立”或缩小板块范围'
          : '请求超时，请稍后重试'
        notes.value = [hint]
      }
      return
    }
    notes.value = [`请求失败: ${msg}`]
  } finally {
    clearTimeout(timeoutId)
    if (refreshAbortController === controller) {
      refreshAbortController = null
    }
    if (requestSeq === refreshRequestSeq) {
      loading.value = false
    }
    if (fullModelSeq === modelRequestSeq) {
      modelLoading.value = false
    }
  }
}

async function refreshModelTop() {
  const requestSeq = ++modelRequestSeq
  if (modelAbortController) {
    modelAbortController.abort()
  }
  const controller = new AbortController()
  modelAbortController = controller
  const timeoutMs = modelIndependent.value ? 70000 : 50000
  let timedOut = false
  const timeoutId = setTimeout(() => {
    timedOut = true
    controller.abort()
  }, timeoutMs)
  modelLoading.value = true

  const applyModelPayload = (data) => {
    modelTop.value = data.model_top || []
    notes.value = data.notes || []
    stats.value = data.stats || {}
    marketOpen.value = data.market_open
    serverTime.value = data.server_time || ''
    scheduleModelWarmupRetry(data)
  }

  const fetchModelPayload = async (signal, modelIndependentFlag) => {
    const url = apiUrl('/api/model-top')
    url.searchParams.set('mode', mode.value)
    url.searchParams.set('provider', provider.value)
    syncSelectedModelKey()
    if (modelKey.value) url.searchParams.set('model', modelKey.value)
    url.searchParams.set('model_independent', String(modelIndependentFlag))
    if (modelSector.value) url.searchParams.set('model_sector', modelSector.value)
    url.searchParams.set('top_n', String(topN.value))
    url.searchParams.set('intraday', 'false')
    const res = await fetch(url, { signal })
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    return await res.json()
  }

  try {
    const data = await fetchModelPayload(controller.signal, modelIndependent.value)
    if (requestSeq !== modelRequestSeq) return
    applyModelPayload(data)
  } catch (e) {
    if (requestSeq !== modelRequestSeq) return
    const msg = (e && e.message) ? String(e.message) : 'request failed'
    const shouldFallback =
      modelIndependent.value && (
        (e && e.name === 'AbortError' && timedOut) ||
        msg.includes('HTTP 504')
      )
    if (shouldFallback) {
      const retryController = new AbortController()
      const retryTimeout = setTimeout(() => {
        retryController.abort()
      }, 50000)
      try {
        const data = await fetchModelPayload(retryController.signal, false)
        if (requestSeq !== modelRequestSeq) return
        modelIndependent.value = false
        applyModelPayload(data)
        return
      } finally {
        clearTimeout(retryTimeout)
      }
    }
  } finally {
    clearTimeout(timeoutId)
    if (modelAbortController === controller) {
      modelAbortController = null
    }
    if (requestSeq === modelRequestSeq) {
      modelLoading.value = false
    }
  }
}

function clearModelWarmupRetry() {
  if (modelWarmupRetryTimer) {
    clearTimeout(modelWarmupRetryTimer)
    modelWarmupRetryTimer = null
  }
}

function shouldRetryModelWarmup(data) {
  if (!modelIndependent.value) return false
  const items = Array.isArray(data?.model_top) ? data.model_top : []
  if (items.length > 0) return false
  const hintNotes = Array.isArray(data?.notes) ? data.notes : []
  return hintNotes.some((note) => {
    const text = String(note || '')
    return text.includes('模型榜单首次生成中') || text.includes('后台刷新中') || text.includes('后台刷新已触发')
  })
}

function scheduleModelWarmupRetry(data) {
  clearModelWarmupRetry()
  if (!shouldRetryModelWarmup(data)) return
  modelWarmupRetryTimer = setTimeout(() => {
    if (viewPage.value !== 'market' || modelLoading.value || loading.value || !modelIndependent.value) return
    refreshModelTop()
  }, 3000)
}

function getTodayCNDate() {
  const now = new Date()
  const utc = now.getTime() + now.getTimezoneOffset() * 60000
  const cn = new Date(utc + 8 * 3600000)
  return cn.toISOString().slice(0, 10)
}

async function loadReviewJournal() {
  const d = reviewDate.value || getTodayCNDate()
  reviewDate.value = d
  reviewError.value = ''
  try {
    const url = apiUrl('/api/review/journal')
    url.searchParams.set('review_date', d)
    const res = await fetch(url)
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    const data = await res.json()
    reviewMessages.value = data.messages || []
    reviewOperations.value = data.operations || []
    reviewCard.value = data.card || {}
    reviewErrorStats.value = data.error_stats_month || []
  } catch (e) {
    reviewError.value = (e && e.message) ? `复盘记录加载失败: ${e.message}` : '复盘记录加载失败'
  }
}

async function sendChat() {
  const text = String(chatInput.value || '').trim()
  if (!text || chatSending.value) return
  chatSending.value = true
  reviewError.value = ''
  try {
    const res = await fetch(apiEndpoint('/api/review/chat'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        date: reviewDate.value,
        text
      })
    })
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    const data = await res.json()
    if (data.error) throw new Error(data.error)
    reviewMessages.value = data.messages || []
    reviewOperations.value = data.operations || []
    reviewCard.value = data.card || {}
    reviewErrorStats.value = data.error_stats_month || []
    chatInput.value = ''
  } catch (e) {
    reviewError.value = (e && e.message) ? `聊天发送失败: ${e.message}` : '聊天发送失败'
  } finally {
    chatSending.value = false
  }
}

async function generateReview() {
  reviewLoading.value = true
  reviewError.value = ''
  try {
    const res = await fetch(apiEndpoint('/api/review'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        mode: mode.value,
        rows: rows.value,
        model_top: modelTop.value,
        notes: notes.value,
        news_summary: newsSummary.value,
        watchlist: watchlist.value,
        review_date: reviewDate.value,
        operations: reviewOperations.value,
        note_text: ''
      })
    })
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    const data = await res.json()
    reviewText.value = data.review || ''
    reviewSource.value = data.source || ''
    reviewCard.value = data.card || reviewCard.value || {}
    monitorConfigPath.value = data.monitor_config || ''
    await loadReviewJournal()
  } catch (e) {
    reviewError.value = (e && e.message) ? `复盘生成失败: ${e.message}` : '复盘生成失败'
  } finally {
    reviewLoading.value = false
  }
}

function onChatInputKeydown(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    sendChat()
  }
}

function startAutoRefresh() {
  if (refreshTimer) {
    clearInterval(refreshTimer)
    refreshTimer = null
  }
  const sec = 5 * 60
  refreshTimer = setInterval(() => {
    if (!isReviewPage.value && marketOpen.value && !loading.value) refreshNow()
  }, sec * 1000)
}

async function openExplain(row) {
  if (!row || !row.symbol) return
  const symbol = row.symbol
  const cached = explainCache.value[symbol]
  explainTitle.value = `${symbol} ${row.name || ''}`.trim()
  explainSource.value = ''
  explainText.value = cached?.text || ''
  explainLevels.value = cached?.levels || null
  explainOpen.value = true
  if (cached) return
  explainLoading.value = true
  try {
    const res = await fetch(apiEndpoint('/api/explain'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ symbol, row })
    })
    const data = await res.json()
    const text = data.explain || data.error || '暂无解释'
    explainCache.value[symbol] = { text, source: data.source || '', levels: data.levels || null }
    explainText.value = text
    explainSource.value = data.source || ''
    explainLevels.value = data.levels || null
  } catch (e) {
    explainText.value = 'AI解释失败'
  } finally {
    explainLoading.value = false
  }
}

onMounted(async () => {
  reviewDate.value = getTodayCNDate()
  const savedViewPage = localStorage.getItem('sqb_view_page')
  if (savedViewPage === 'market' || savedViewPage === 'review') {
    viewPage.value = savedViewPage
  }
  const savedMode = localStorage.getItem('sqb_mode')
  if (savedMode === 'watchlist' || savedMode === 'all') {
    mode.value = savedMode
  }
  const savedProvider = localStorage.getItem('sqb_provider')
  if (savedProvider === 'biying' || savedProvider === 'tencent' || savedProvider === 'auto') {
    provider.value = savedProvider
  }
  const savedModel = localStorage.getItem('sqb_model')
  const savedModelIndependent = localStorage.getItem('sqb_model_independent')
  const savedModelSector = localStorage.getItem('sqb_model_sector')
  if (savedModelIndependent === '1' || savedModelIndependent === 'true') {
    modelIndependent.value = true
  }
  if (savedModelSector) {
    modelSector.value = savedModelSector
  }
  await fetchModelOptions()
  if (savedModel) {
    const exists = modelOptions.value.some((x) => String(x.key || '') === savedModel)
    if (exists) modelKey.value = savedModel
  }
  await fetchWatchlist()
  await refreshNow()
  await loadReviewJournal()
  setTimeout(() => {
    fetchNews()
  }, 0)
  startAutoRefresh()
})

watch(mode, (val) => {
  localStorage.setItem('sqb_mode', val)
})

watch(provider, (val) => {
  localStorage.setItem('sqb_provider', val)
})

watch(modelKey, (val) => {
  localStorage.setItem('sqb_model', String(val || ''))
})

watch(modelIndependent, (val) => {
  localStorage.setItem('sqb_model_independent', val ? '1' : '0')
})

watch(modelSector, (val) => {
  localStorage.setItem('sqb_model_sector', String(val || ''))
})

watch(viewPage, async (val) => {
  localStorage.setItem('sqb_view_page', val)
  if (val === 'market' && !loading.value) {
    await refreshNow()
  }
})

watch(reviewDate, async () => {
  await loadReviewJournal()
})

onUnmounted(() => {
  if (refreshTimer) {
    clearInterval(refreshTimer)
    refreshTimer = null
  }
  clearModelWarmupRetry()
  if (refreshAbortController) {
    refreshAbortController.abort()
    refreshAbortController = null
  }
  if (modelAbortController) {
    modelAbortController.abort()
    modelAbortController = null
  }
})

</script>

<style scoped>
:root {
  color-scheme: light;
}

.app {
  font-family: 'IBM Plex Sans', 'PingFang SC', sans-serif;
  background: radial-gradient(1200px 600px at 20% -10%, #f2f0ff, #f7f6f2 35%, #f9fafb 70%);
  min-height: 100vh;
  padding: 24px;
  color: #1b1f2a;
}

.hero {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 16px;
  padding: 18px 22px;
  background: linear-gradient(135deg, #ffffff, #f6f0ff 60%, #eef5ff);
  border-radius: 16px;
  box-shadow: 0 8px 24px rgba(17, 24, 39, 0.08);
}

.hero h1 {
  font-family: 'ZCOOL XiaoWei', serif;
  font-size: 28px;
  margin: 0;
}

.hero p {
  margin: 6px 0 0;
  color: #5b6472;
}

.status {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.chip {
  padding: 6px 10px;
  border-radius: 999px;
  font-size: 12px;
  background: #e8eefc;
  color: #224;
  display: inline-flex;
  align-items: center;
  gap: 6px;
}

.chip.open {
  background: #e1f7e8;
  color: #135c2f;
}

.chip.closed {
  background: #ffe8e8;
  color: #7c1d1d;
}

.chip.muted {
  background: #f0f0f0;
  color: #6b7280;
}

.view-tabs {
  margin-top: 14px;
  display: flex;
  gap: 10px;
}

.tab-btn {
  border: 1px solid #d1d5db;
  background: #fff;
  color: #111827;
  border-radius: 10px;
  padding: 8px 14px;
  cursor: pointer;
  font-weight: 600;
}

.tab-btn.active {
  background: #1f2937;
  color: #fff;
  border-color: #1f2937;
}

.controls {
  margin: 18px 0;
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 12px;
}

.control {
  background: #ffffff;
  border-radius: 12px;
  padding: 12px;
  box-shadow: 0 6px 18px rgba(17, 24, 39, 0.06);
}

.control label {
  display: block;
  font-size: 12px;
  color: #6b7280;
  margin-bottom: 6px;
}

.control select,
.control input[type='number'],
.control input[type='text'] {
  width: 100%;
  padding: 6px 8px;
  border-radius: 8px;
  border: 1px solid #e5e7eb;
  box-sizing: border-box;
}

.control.actions {
  display: flex;
  align-items: flex-end;
}

.control.actions button {
  width: 100%;
  padding: 10px;
  border: none;
  border-radius: 10px;
  background: #1f2937;
  color: #fff;
  cursor: pointer;
}

.model-switch {
  grid-column: span 2;
}

.model-buttons {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.model-buttons .ghost.small.active {
  background: #dbeafe;
  border-color: #93c5fd;
  color: #1e3a8a;
}

.grid {
  display: grid;
  grid-template-columns: minmax(0, 2.6fr) minmax(280px, 1fr);
  gap: 16px;
  align-items: start;
}

.col {
  display: flex;
  flex-direction: column;
  gap: 16px;
  align-items: stretch;
}

.panel {
  background: #ffffff;
  border-radius: 16px;
  padding: 16px;
  border: 1px solid #e7eaf1;
  box-shadow: 0 10px 26px rgba(17, 24, 39, 0.06);
}

.review-page {
  margin-top: 16px;
}

.review-layout {
  display: grid;
  grid-template-columns: minmax(0, 1.7fr) minmax(320px, 1fr);
  gap: 16px;
}

.review-chat-panel,
.review-card-panel {
  min-height: 560px;
}

.chat-list {
  border: 1px solid #e6e9f0;
  border-radius: 10px;
  background: #fbfcfe;
  height: 360px;
  overflow: auto;
  padding: 10px;
  margin-bottom: 10px;
}

.chat-item {
  margin-bottom: 10px;
  border-radius: 10px;
  padding: 8px 10px;
  border: 1px solid #e5e7eb;
}

.chat-item.user {
  background: #eef6ff;
  border-color: #cfe1ff;
}

.chat-item.assistant {
  background: #f8fafc;
}

.chat-role {
  font-size: 11px;
  color: #6b7280;
  margin-bottom: 4px;
}

.chat-text {
  white-space: pre-wrap;
  line-height: 1.55;
  color: #1f2937;
}

.chat-input {
  width: 100%;
  box-sizing: border-box;
  border: 1px solid #d5dbe8;
  border-radius: 10px;
  padding: 8px 10px;
  font-size: 13px;
  margin-bottom: 8px;
}

.review-actions {
  display: flex;
  justify-content: flex-start;
  margin-bottom: 10px;
}

.review-actions.single-action {
  margin-top: 10px;
  margin-bottom: 10px;
}

.ghost.primary {
  background: #1f2937;
  color: #fff;
  border-color: #1f2937;
}

.review-text {
  line-height: 1.7;
  color: #243041;
  white-space: pre-wrap;
}

.review-date-row {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 10px;
}

.review-date-row label {
  font-size: 12px;
  color: #6b7280;
  min-width: 56px;
}

.review-date-row input {
  border: 1px solid #d5dbe8;
  border-radius: 8px;
  padding: 6px 8px;
}

.review-op-form {
  display: grid;
  grid-template-columns: 110px 86px 96px 96px 96px 88px;
  gap: 8px;
  margin-bottom: 8px;
}

.review-op-form-note {
  grid-template-columns: 1fr;
}

.review-op-form input,
.review-op-form select {
  border: 1px solid #d5dbe8;
  border-radius: 8px;
  padding: 6px 8px;
  font-size: 12px;
  box-sizing: border-box;
}

.review-op-list {
  border: 1px solid #e6e9f0;
  border-radius: 10px;
  background: #fafcff;
  max-height: 180px;
  overflow: auto;
  margin-bottom: 10px;
}

.review-op-item {
  display: grid;
  grid-template-columns: 78px 52px 68px 64px 72px 1fr 58px;
  gap: 8px;
  align-items: center;
  border-bottom: 1px solid #eef1f6;
  padding: 6px 8px;
  font-size: 12px;
}

.review-op-item:last-child {
  border-bottom: 0;
}

.review-op-item .reason {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.review-note-box {
  margin-bottom: 10px;
}

.review-note-box label {
  display: block;
  font-size: 12px;
  color: #6b7280;
  margin-bottom: 6px;
}

.review-note-box textarea {
  width: 100%;
  box-sizing: border-box;
  border: 1px solid #d5dbe8;
  border-radius: 8px;
  padding: 8px;
  font-size: 13px;
  resize: vertical;
  min-height: 84px;
}

.card-block {
  border: 1px solid #e6e9f0;
  border-radius: 10px;
  background: #fafcff;
  padding: 10px;
  margin-bottom: 10px;
}

.card-title {
  font-weight: 600;
  font-size: 13px;
  margin-bottom: 6px;
}

.card-sub {
  font-size: 12px;
  color: #6b7280;
  margin-top: 6px;
  margin-bottom: 4px;
}

.card-op-item {
  display: grid;
  grid-template-columns: 72px 50px 70px 68px 1fr 1fr;
  gap: 8px;
  border-bottom: 1px solid #eef1f6;
  padding: 6px 0;
  font-size: 12px;
  align-items: center;
}

.card-op-item:last-child {
  border-bottom: none;
}

.card-line {
  font-size: 13px;
  color: #1f2937;
  line-height: 1.5;
  margin-bottom: 4px;
}

.card-tag {
  display: inline-block;
  margin-top: 6px;
  padding: 2px 8px;
  border-radius: 999px;
  background: #fee2e2;
  color: #991b1b;
  font-size: 12px;
}

.panel-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
  padding-bottom: 8px;
  border-bottom: 1px solid #eef0f4;
}

.panel-header h2 {
  margin: 0;
  font-size: 18px;
}

.meta {
  font-size: 12px;
  color: #6b7280;
}

.table-wrap {
  position: relative;
  max-height: 560px;
  overflow: auto;
  border: 1px solid #e6e9f0;
  border-radius: 12px;
  background: #fff;
}

.table-wrap table {
  width: 100%;
  border-collapse: collapse;
  min-width: 1100px;
  table-layout: fixed;
}

.table-wrap th,
.table-wrap td {
  padding: 8px 10px;
  border-bottom: 1px solid #eef0f4;
  font-size: 13px;
  vertical-align: middle;
}


.table-wrap th {
  position: sticky;
  top: 0;
  background: #f8fafc;
  font-weight: 600;
  letter-spacing: 0.2px;
  text-align: left;
}

.table-wrap tbody tr:nth-child(even) {
  background: #fbfcfe;
}

.table-wrap tbody tr:hover {
  background: #f1f5f9;
}

.table-wrap th.sortable {
  cursor: pointer;
  user-select: none;
}

.table-wrap th.sortable:hover {
  background: #f3f4f6;
}

.arrow {
  font-size: 11px;
  color: #9ca3af;
  margin-left: 4px;
}

.mono {
  font-family: 'IBM Plex Sans', monospace;
  letter-spacing: 0.5px;
}

.badge {
  padding: 3px 6px;
  border-radius: 6px;
  font-weight: 600;
  font-size: 12px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 44px;
}

.badge.buy { background: #e0f7f0; color: #0f766e; }
.badge.watch { background: #fef3c7; color: #92400e; }
.badge.avoid { background: #fde2e2; color: #b91c1c; }
.badge.hot { background: #ffe4d6; color: #9a3412; }

.up { color: #b91c1c; }
.down { color: #1d4ed8; }

.reason {
  max-width: 320px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.overlay {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(255, 255, 255, 0.75);
  font-weight: 600;
}

.overlay.empty {
  color: #9ca3af;
}

.side .list {
  list-style: none;
  padding: 0;
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.side .list li {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px;
  border-radius: 10px;
  background: #f8fafc;
}

.side .title {
  display: flex;
  gap: 8px;
  font-weight: 600;
}

.side .sub {
  display: flex;
  gap: 12px;
  color: #6b7280;
  font-size: 12px;
}

.ghost {
  border: 1px solid #cbd5f5;
  background: #fff;
  border-radius: 8px;
  padding: 6px 8px;
  cursor: pointer;
}

.ghost.small {
  font-size: 12px;
  padding: 4px 8px;
  border-radius: 999px;
  border-color: #e5e7eb;
  background: #f8fafc;
}

.ai {
  text-align: center;
}

.col-symbol { width: 78px; }
.col-name { width: 120px; }
.col-action { width: 62px; text-align: center; }
.col-sector { width: 96px; }
.col-score { width: 84px; text-align: right; }
.col-pct { width: 86px; text-align: right; }
.col-reason { width: 200px; }
.col-ai { width: 92px; text-align: center; }

th.col-score,
th.col-pct {
  text-align: right;
}


.watchlist .chips {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.watchlist-add {
  display: flex;
  gap: 8px;
  margin-bottom: 12px;
}

.watchlist-add input {
  flex: 1;
  padding: 8px 10px;
  border-radius: 8px;
  border: 1px solid #e5e7eb;
}

.watchlist-add button {
  padding: 8px 12px;
  border: none;
  border-radius: 8px;
  background: #1f2937;
  color: #fff;
  cursor: pointer;
}

.news-controls {
  display: flex;
  gap: 8px;
  margin-bottom: 10px;
}

.summary-card {
  background: #f3f5f8;
  border-radius: 12px;
  padding: 12px 14px;
  margin-bottom: 12px;
}

.summary-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
}

.summary-label {
  font-size: 12px;
  color: #6b7280;
  margin-bottom: 4px;
}

.summary-value {
  font-weight: 600;
  color: #1f2937;
}

.summary-score {
  display: inline-block;
  margin-left: 6px;
  padding: 2px 8px;
  border-radius: 999px;
  background: #e0e7ff;
  font-size: 12px;
  color: #1d4ed8;
}

.summary-full {
  grid-column: 1 / -1;
}

.summary-text {
  color: #374151;
  font-size: 13px;
  line-height: 1.5;
}

.news-controls select {
  flex: 1;
  padding: 6px 8px;
  border-radius: 8px;
  border: 1px solid #e5e7eb;
}

.news-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
  max-height: 260px;
  overflow: auto;
}

.news-item {
  display: block;
  padding: 10px;
  border-radius: 10px;
  background: #f8fafc;
  color: inherit;
  text-decoration: none;
  border: 1px solid #eef0f4;
}

.news-title {
  font-size: 13px;
  font-weight: 600;
  margin-bottom: 4px;
}

.news-meta {
  display: flex;
  justify-content: space-between;
  font-size: 12px;
  color: #6b7280;
}

.sentiment {
  padding: 2px 6px;
  border-radius: 999px;
  font-size: 11px;
  background: #eef2ff;
  color: #3730a3;
}

.sentiment.pos {
  background: #dcfce7;
  color: #166534;
}

.sentiment.neg {
  background: #fee2e2;
  color: #991b1b;
}

.sentiment.neu {
  background: #e5e7eb;
  color: #374151;
}

.muted { color: #9ca3af; }

.context {
  position: fixed;
  z-index: 1000;
  background: #fff;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  padding: 6px;
  box-shadow: 0 10px 22px rgba(0,0,0,0.12);
}

.context button {
  border: none;
  background: #1f2937;
  color: #fff;
  padding: 6px 10px;
  border-radius: 6px;
  cursor: pointer;
}

.note {
  font-size: 12px;
  color: #6b7280;
  margin-bottom: 8px;
}

.modal {
  position: fixed;
  inset: 0;
  background: rgba(15, 23, 42, 0.35);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 50;
}

.modal-card {
  width: min(680px, 90vw);
  background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
  border-radius: 16px;
  padding: 16px;
  box-shadow: 0 20px 50px rgba(15, 23, 42, 0.2);
}

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
  margin-bottom: 12px;
}

.modal-title {
  font-weight: 600;
  font-size: 16px;
}

.modal-body {
  line-height: 1.6;
  color: #1f2937;
}

.level-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(120px, 1fr));
  gap: 12px;
  margin-bottom: 12px;
}

.level {
  background: #ffffff;
  border: 1px solid #e5e7eb;
  border-radius: 12px;
  padding: 10px;
  box-shadow: 0 6px 14px rgba(15, 23, 42, 0.06);
}

.level .label {
  font-size: 12px;
  color: #6b7280;
}

.level .value {
  font-size: 18px;
  font-weight: 600;
  color: #0f172a;
  margin-top: 4px;
}

.footnote {
  margin-top: 10px;
  font-size: 12px;
  color: #94a3b8;
}

@media (max-width: 1100px) {
  .controls { grid-template-columns: repeat(2, minmax(140px, 1fr)); }
  .grid { grid-template-columns: 1fr; }
  .review-layout { grid-template-columns: 1fr; }
  .review-op-form { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
  .review-op-item { grid-template-columns: 72px 48px 64px 60px 68px 1fr 56px; }
  .card-op-item { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
}
</style>

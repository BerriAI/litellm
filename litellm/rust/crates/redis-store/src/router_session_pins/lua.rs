/// `_READ_PIN_SCRIPT`, byte for byte, so pods on either stack share one script SHA.
pub(super) const READ_PIN: &str = r#"
local current = redis.call('GET', KEYS[1])
if current ~= false then
  redis.call('EXPIRE', KEYS[1], ARGV[1])
end
return current
"#;

/// `_CLAIM_PIN_SCRIPT`, byte for byte.
pub(super) const CLAIM_PIN: &str = r#"
local current = redis.call('GET', KEYS[1])
if current == false then
  redis.call('SET', KEYS[1], ARGV[1], 'EX', ARGV[2])
  return ARGV[1]
end
if ARGV[3] then
  local decoded, stored = pcall(cjson.decode, current)
  if ARGV[4] then
    local legacy = current
    if decoded and type(stored) == 'string' then legacy = stored end
    for _, eligible in ipairs(cjson.decode(ARGV[3])) do
      if eligible['model_id'] == legacy then
        redis.call('EXPIRE', KEYS[1], ARGV[2])
        return current
      end
    end
  end
  if decoded and type(stored) == 'table' then
    for _, eligible in ipairs(cjson.decode(ARGV[3])) do
      local matches = true
      for key, value in pairs(eligible) do
        if stored[key] ~= value then matches = false; break end
      end
      for key, _ in pairs(stored) do
        if eligible[key] == nil then matches = false; break end
      end
      if matches then
        redis.call('EXPIRE', KEYS[1], ARGV[2])
        return current
      end
    end
  end
  redis.call('SET', KEYS[1], ARGV[1], 'EX', ARGV[2])
  return ARGV[1]
end
if current == ARGV[1] then
  redis.call('EXPIRE', KEYS[1], ARGV[2])
end
return current
"#;

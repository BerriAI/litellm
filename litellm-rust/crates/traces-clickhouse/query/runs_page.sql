page AS (
SELECT * EXCEPT (root_input, search_root_status, search_has_error, matched_attributes),
       multiIf({sort_key:String} = 'duration_ms', toInt64(least(duration_ns, toUInt64(9223372036854775807))),
               {sort_key:String} = 'span_count', toInt64(least(span_count, toUInt64(9223372036854775807))),
               {sort_key:String} = 'error_count', toInt64(least(error_count, toUInt64(9223372036854775807))),
               {sort_key:String} = 'trace_ref', toInt64(0),
               start_ms) AS sort_value
FROM runs
WHERE {has_cursor:UInt8} = 0
   OR if({descending:UInt8} = 1,
         (sort_value, trace_ref) < ({cursor_value:Int64}, {cursor_ref:String}),
         (sort_value, trace_ref) > ({cursor_value:Int64}, {cursor_ref:String}))
ORDER BY if({descending:UInt8} = 1, sort_value, 0) DESC, if({descending:UInt8} = 1, trace_ref, '') DESC,
         sort_value, trace_ref
LIMIT {limit:UInt32}
)

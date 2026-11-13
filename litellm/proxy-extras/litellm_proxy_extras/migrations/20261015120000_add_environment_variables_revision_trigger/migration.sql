SET LOCAL lock_timeout = '10s';

CREATE SEQUENCE IF NOT EXISTS "litellm_environment_variables_revision_seq";
GRANT USAGE ON SEQUENCE "litellm_environment_variables_revision_seq" TO PUBLIC;

CREATE OR REPLACE FUNCTION litellm_bump_environment_variables_revision() RETURNS TRIGGER AS $$
DECLARE
  next_revision BIGINT := nextval(format('%I.litellm_environment_variables_revision_seq', TG_TABLE_SCHEMA)::regclass);
BEGIN
  IF TG_OP = 'UPDATE' THEN
    NEW."reload_revision" := GREATEST(next_revision, OLD."reload_revision" + 1);
  ELSE
    NEW."reload_revision" := next_revision;
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS "litellm_environment_variables_revision" ON "LiteLLM_Config";
CREATE TRIGGER "litellm_environment_variables_revision"
BEFORE INSERT OR UPDATE ON "LiteLLM_Config"
FOR EACH ROW WHEN (NEW."param_name" = 'environment_variables')
EXECUTE PROCEDURE litellm_bump_environment_variables_revision();
